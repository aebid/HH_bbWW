# This is a combination of several steps
# 1. Open and hadd the separate parity files per mass point
# 2. Take the hadded files and fit the separate background distributions as well as sum background
# 3. Take the fit, and original hists and combine into a single new file with the correct naming scheme
# 4. Rebin the combined shapes to be something reasonable for combine, and save to a new file for combine
# 5. Quickly measure limits of the shape files

import os
import ROOT
import math
import shutil
from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.envToJson import get_cmsenv
import uproot
import json
import matplotlib.pyplot as plt

cmssw_env = get_cmsenv(cmssw_path=os.getenv("FLAF_CMSSW_BASE"))


def hadd_parity_files(
    filepath_resolved, filepath_boosted, histname, masslist, catlist, output_dir, bkg_list
):
    # For each mass, load the validaiton files for each parity and combine all background into one distribution and signal into another distribution. Then make a plot of the two distributions and save it to disk.

    # bkg_list = ["TT", "DY", "ST", "SMHiggs", "VV", "Other"]
    sig_list = ["Signal"]

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, "hadd_m{mass}_{cat}.root")

    for cat in catlist:
        filepath = filepath_boosted if cat == "boosted" else filepath_resolved
        for mass in masslist:
            # Load the 4 parity files and combine signal and all backgrounds
            signal = []
            background = []
            hist_list = []
            bkg_dict = {}
            for proc in bkg_list:
                bkg_dict[proc] = []
            for par in [0, 1, 2, 3]:
                file_name = filepath.format(par=par, mass=mass, cat=cat)
                if not os.path.exists(file_name):
                    print(f"File {file_name} does not exist, skipping.")
                    continue
                # Load the file and extract the signal and background distributions
                f = ROOT.TFile(file_name)

                for proc in bkg_list:
                    hist_list.append(
                        f.Get(histname.format(mass=mass, proc=proc)).Clone()
                    )
                    hist = hist_list[-1]
                    hist.SetDirectory(0)  # Detach the histogram from the file
                    if hist is None:
                        print(
                            f"Histogram {histname.format(mass=mass, proc=proc)} not found in file {file_name}, skipping."
                        )
                        continue
                    background.append(hist)
                    bkg_dict[proc].append(hist)

                for proc in sig_list:
                    hist_list.append(
                        f.Get(histname.format(mass=mass, proc=proc)).Clone()
                    )
                    hist = hist_list[-1]
                    hist.SetDirectory(0)  # Detach the histogram from the file
                    if hist is None:
                        print(
                            f"Histogram {histname.format(mass=mass, proc=proc)} not found in file {file_name}, skipping."
                        )
                        continue
                    signal.append(hist)

            # Combine the signal and background histograms
            if len(signal) == 0 or len(background) == 0:
                print(
                    f"No valid signal or background histograms found for mass {mass} and category {cat}, skipping."
                )
                continue
            signal_combined = signal[0].Clone()
            signal_combined.SetTitle("Signal (combined)")
            signal_combined.SetName("Signal (combined)")
            for hist in signal[1:]:
                signal_combined.Add(hist)
            background_combined = background[0].Clone()
            background_combined.SetTitle("Background (combined)")
            background_combined.SetName("Background (combined)")
            for hist in background[1:]:
                background_combined.Add(hist)

            bkg_combined_dict = {}
            for bkg in bkg_list:
                bkg_combined_dict[bkg] = bkg_dict[bkg][0].Clone()
                bkg_combined_dict[bkg].SetTitle(f"{bkg} (combined)")
                bkg_combined_dict[bkg].SetName(f"{bkg} (combined)")
                for hist in bkg_dict[bkg][1:]:
                    bkg_combined_dict[bkg].Add(hist)

            # Save the new hadd histograms to a file, and create a canvas with the two distributions overlaid and save it to disk
            # output_file = ROOT.TFile(os.path.join(output_dir, f"hadd_m{mass}_{cat}.root"), "RECREATE")
            output_file = ROOT.TFile(
                output_filepath.format(mass=mass, cat=cat), "RECREATE"
            )
            signal_combined.Write("signal")
            background_combined.Write("background")
            for bkg in bkg_list:
                bkg_combined_dict[bkg].Write(f"{bkg}")

            # Create a canvas and plot the two distributions
            # Normalize the two distributions to the same area
            c = ROOT.TCanvas(f"c_m{mass}_{cat}", f"m{mass} {cat}", 800, 600)
            signal_combined.SetLineColor(ROOT.kRed)
            background_combined.SetLineColor(ROOT.kBlue)
            signal_combined.SetLineWidth(2)
            background_combined.SetLineWidth(2)
            signal_combined.Draw()
            background_combined.Draw("SAME")
            signal_combined.GetYaxis().SetRangeUser(
                0,
                1.2
                * max(signal_combined.GetMaximum(), background_combined.GetMaximum()),
            )
            c.BuildLegend()
            # Save canvas to same output_file as the histograms
            c.Write("plot")

            # Make a not-normalized canvas with the individual background contributions (not stacked) and the signal
            c2 = ROOT.TCanvas(
                f"c2_m{mass}_{cat}", f"m{mass} {cat} (individual bkg)", 800, 600
            )
            signal_combined.SetLineColor(ROOT.kRed)
            signal_combined.SetLineWidth(2)
            signal_combined.Draw()
            for bkg in bkg_list:
                # Cycle through some colors for the individual background contributions
                color = ROOT.kBlue + 10 * bkg_list.index(bkg)
                bkg_combined_dict[bkg].SetLineColor(color)
                bkg_combined_dict[bkg].SetLineWidth(2)
                bkg_combined_dict[bkg].Draw("SAME")
            signal_combined.GetYaxis().SetRangeUser(
                0,
                1.2
                * max(
                    signal_combined.GetMaximum(),
                    max(bkg_combined_dict[bkg].GetMaximum() for bkg in bkg_list),
                ),
            )
            c2.BuildLegend()
            c2.Write("plot_individual_bkg")

            if signal_combined.Integral() > 0:
                signal_combined.Scale(1.0 / signal_combined.Integral())
                background_combined.Scale(1.0 / background_combined.Integral())
            signal_combined.GetYaxis().SetRangeUser(
                0,
                1.2
                * max(signal_combined.GetMaximum(), background_combined.GetMaximum()),
            )
            c.Write("plot_normalized")

            output_file.Close()

    return output_filepath


def fit_hadded_shapes(filepath, masslist, catlist, output_dir, bkg_list):
    # load the hadd_m{mass}_{cat}.root files, load the background distribution, and create a fit

    # Set minimizer to MINUIT2
    ROOT.Math.MinimizerOptions.SetDefaultMinimizer("Minuit2", "Migrad")

    # Configuration
    use_double_crystal_ball = (
        False  # Set to False for single-sided crystal ball, True for double-sided
    )
    # fit_option = "crystal_ball"
    # fit_option = "double_crystal_ball"
    # fit_option = "crystal_ball_expo"
    fit_option = "crystal_ball_gaus"

    hists_to_fit = [ "signal", "background" ] + bkg_list
    # hists_to_fit = ["background", "TT", "DY", "ST", "SMHiggs", "VV", "Other", "signal"]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    output_filepath = os.path.join(output_dir, "fit_hist_m{mass}_{cat}.root")

    # Loop over the mass and category, load the background distribution, and create a fit
    for mass in masslist:
        os.makedirs(os.path.join(output_dir, f"m{mass}"), exist_ok=True)
        for cat in catlist:
            file_name = filepath.format(mass=mass, cat=cat)
            if not os.path.exists(file_name):
                print(f"File {file_name} does not exist, skipping.")
                continue
            f = ROOT.TFile(file_name)
            # output_file = ROOT.TFile(os.path.join(output_dir, f"fit_hist_m{mass}_{cat}.root"), "RECREATE")
            output_file = ROOT.TFile(
                output_filepath.format(mass=mass, cat=cat), "RECREATE"
            )

            # Save all original histograms from the input file to the output file
            print("Copying original histograms from input file...")
            for key in f.GetListOfKeys():
                obj = f.Get(key.GetName())
                if isinstance(obj, ROOT.TH1):
                    output_file.cd()
                    obj_clone = obj.Clone()
                    obj_clone.Write(key.GetName())
                    obj_clone.Delete()

                    # Save total background as 'data_obs' for combine
                    if key.GetName() == "background":
                        data_obs = obj.Clone("data_obs")
                        data_obs.Write("data_obs")
                        data_obs.Delete()

            for hist_idx, histname in enumerate(hists_to_fit):
                print(
                    f"Going to fit histogram: {histname} for mass {mass} and category {cat}"
                )
                hist = f.Get(histname)
                if hist is None:
                    print(
                        f"Histogram {histname} not found in file {file_name}, skipping."
                    )
                    continue

                # Clone the histogram to keep it valid after closing the file
                hist = hist.Clone()

                # Validate histogram
                if hist.GetEntries() == 0:
                    print(f"Histogram {histname} is empty, skipping.")
                    hist.Delete()
                    continue

                # Check for NaN or invalid values
                has_valid_data = False
                for i in range(1, hist.GetNbinsX() + 1):
                    if hist.GetBinContent(i) > 0:
                        has_valid_data = True
                        break
                if not has_valid_data:
                    print(f"Histogram {histname} has no valid bin contents, skipping.")
                    hist.Delete()
                    continue

                # Create a fit to the background distribution using a polynomial of degree 3
                # Use unique names for each fit function to avoid ROOT object registry conflicts
                fit_name = f"fit_{mass}_{cat}_{hist_idx}_{histname}"

                # Set initial parameters with better validation
                mean = hist.GetMean()
                rms = hist.GetRMS()
                x_min = hist.GetXaxis().GetXmin()
                x_max = hist.GetXaxis().GetXmax()
                x_range = x_max - x_min
                if x_range <= 0:
                    print(
                        f"Histogram {histname} has invalid x range [{x_min}, {x_max}], skipping."
                    )
                    hist.Delete()
                    continue

                peak_bin = hist.GetMaximumBin()
                peak_center = hist.GetXaxis().GetBinCenter(peak_bin)
                if abs(peak_center - mean) > 0.5 * x_range:
                    mean = peak_center

                # Check for reasonable RMS
                if rms <= 0:
                    rms = max(0.5, x_range * 0.1)
                    print(
                        f"Warning: RMS for {histname} is invalid, using default value of {rms:.1f}"
                    )

                fit_lo = x_min
                fit_hi = x_max
                sigma_min = max(0.01, x_range * 0.01)
                sigma_max = max(1.0, x_range * 0.5)
                alpha_left_init = 1.5
                alpha_right_init = 1.5
                n_init = 2.0

                if fit_option == "double_crystal_ball":
                    mean = peak_center
                    rms = hist.GetRMS()
                    sigma_init = rms if rms > 0 else x_range * 0.1
                    sigma_init = max(sigma_min, min(sigma_init, sigma_max))

                    alpha_left_init = 1.5
                    alpha_right_init = 1.5
                    n_left_init = 2.0
                    n_right_init = 2.0

                    dscb_expr = (
                        "[0] * ("
                        "(((x-[1])/[2]) < -[3]) ? "
                        "(pow([4]/[3],[4]) * exp(-0.5*[3]*[3]) * pow([4]/[3] - [3] - (x-[1])/[2], -[4])) : "
                        "((((x-[1])/[2]) > [5]) ? "
                        "(pow([6]/[5],[6]) * exp(-0.5*[5]*[5]) * pow([6]/[5] - [5] + (x-[1])/[2], -[6])) : "
                        "exp(-0.5*((x-[1])/[2])*((x-[1])/[2])) ) )"
                    )

                    fit = ROOT.TF1(fit_name, dscb_expr, fit_lo, fit_hi)
                    fit.SetNpx(1000)
                    fit.SetParameters(
                        hist.GetMaximum(),  # amplitude
                        mean,  # mean
                        sigma_init,  # sigma
                        alpha_left_init,  # alphaL
                        n_left_init,  # nL
                        alpha_right_init,  # alphaR
                        n_right_init,  # nR
                    )

                    fit.SetParLimits(0, 0.0, max(1.0, hist.GetMaximum() * 50))
                    fit.SetParLimits(1, fit_lo, fit_hi)
                    fit.SetParLimits(2, sigma_min, sigma_max)
                    fit.SetParLimits(3, 0.1, 10.0)
                    fit.SetParLimits(4, 1.01, 50.0)
                    fit.SetParLimits(5, 0.1, 10.0)
                    fit.SetParLimits(6, 1.01, 50.0)

                    # validate initial evaluation
                    bad_eval = False
                    for i in range(1, hist.GetNbinsX() + 1):
                        x = hist.GetXaxis().GetBinCenter(i)
                        y = fit.Eval(x)
                        if not math.isfinite(y):
                            print(
                                f"Bad initial function eval for {histname}: x={x}, y={y}"
                            )
                            bad_eval = True
                            break

                    if bad_eval:
                        fit.Delete()
                        hist.Delete()
                        continue

                elif fit_option == "crystal_ball_expo":
                    # Crystal Ball LEFT + Exponential RIGHT
                    # Parameters: [0]=amplitude, [1]=mean, [2]=sigma, [3]=alphaL, [4]=nL, [5]=alphaR, [6]=betaR
                    # Left: power-law tail (standard CB)
                    # Right: exp(-betaR * (x-mu)/sigma) starting at alphaR*sigma from mean

                    cb_exp_expr = (
                        "[0] * ("
                        "(((x-[1])/[2]) < -abs([3])) ? "
                        # Left power-law tail
                        "(pow(abs([4])/abs([3]), abs([4])) * exp(-0.5*[3]*[3]) * "
                        "pow(abs([4])/abs([3]) - abs([3]) - (x-[1])/[2], -abs([4]))) : "
                        # Right exponential tail
                        "((((x-[1])/[2]) > abs([5])) ? "
                        "(exp(-0.5*[5]*[5]) * exp(-abs([6]) * ((x-[1])/[2] - abs([5])))) : "
                        # Gaussian core
                        "exp(-0.5*((x-[1])/[2])*((x-[1])/[2])) ) )"
                    )

                    fit = ROOT.TF1(fit_name, cb_exp_expr, fit_lo, fit_hi)
                    fit.SetNpx(1000)

                    fit.SetParameters(
                        hist.GetMaximum(),  # [0] amplitude
                        peak_center,  # [1] mean
                        rms * 0.7,  # [2] sigma (start narrower)
                        1.5,  # [3] alphaL
                        3.0,  # [4] nL
                        1.2,  # [5] alphaR (transition point)
                        2.0,  # [6] betaR (exponential slope)
                    )

                    fit.SetParLimits(0, 0, hist.GetMaximum() * 10)
                    fit.SetParLimits(1, fit_lo, fit_hi)
                    fit.SetParLimits(2, sigma_min, sigma_max)
                    fit.SetParLimits(3, 0.1, 10.0)  # alphaL
                    fit.SetParLimits(4, 1.01, 50.0)  # nL
                    fit.SetParLimits(5, 0.1, 5.0)  # alphaR
                    fit.SetParLimits(6, 0.1, 20.0)  # betaR (larger = faster falloff)

                elif fit_option == "crystal_ball_gaus":
                    # Narrow Gaussian + Wide left-sided Crystal Ball
                    gauss_plus_cb_expr = (
                        "[0] * exp(-0.5*((x-[1])/[2])^2) + "
                        "[3] * ("
                        "(((x-[4])/[5]) < -abs([6])) ? "
                        "(pow(abs([7])/abs([6]), abs([7])) * exp(-0.5*[6]*[6]) * "
                        "pow(abs([7])/abs([6]) - abs([6]) - (x-[4])/[5], -abs([7]))) : "
                        "exp(-0.5*((x-[4])/[5])^2) )"
                    )

                    fit = ROOT.TF1(fit_name, gauss_plus_cb_expr, fit_lo, fit_hi)
                    fit.SetNpx(1000)

                    fit.SetParameters(
                        hist.GetMaximum() * 0.5,  # [0] amp_gauss (narrow peak)
                        peak_center,  # [1] mean_gauss
                        0.5,  # [2] sigma_gauss  ← KEY: much narrower than RMS
                        hist.GetMaximum() * 0.3,  # [3] amp_cb (wide + left tail)
                        peak_center,  # [4] mean_cb
                        1.5,  # [5] sigma_cb
                        1.5,  # [6] alpha_cb
                        3.0,  # [7] n_cb
                    )

                    fit.SetParLimits(0, 0, hist.GetMaximum() * 5)
                    fit.SetParLimits(1, peak_center - 1, peak_center + 1)
                    fit.SetParLimits(2, 0.2, 1.0)  # NARROW
                    fit.SetParLimits(3, 0, hist.GetMaximum() * 5)
                    fit.SetParLimits(4, peak_center - 2, peak_center + 2)
                    fit.SetParLimits(5, 0.8, 4.0)  # WIDE
                    fit.SetParLimits(6, 0.3, 5.0)
                    fit.SetParLimits(7, 1.01, 30.0)

                    fit_result = hist.Fit(fit, "RS")

                else:
                    # Single-sided crystal ball: [0] amplitude, [1] mean, [2] sigma, [3] alpha, [4] n
                    fit = ROOT.TF1(fit_name, "crystalball", fit_lo, fit_hi)
                    fit.SetParameters(hist.GetMaximum(), mean, rms, 2.0, 2.0)
                    print(
                        f"Single-sided crystal ball initialized: amplitude={hist.GetMaximum():.2f}, mean={mean:.4f}, rms={rms:.4f}, "
                        f"alpha={alpha_right_init:.2f}, n={n_init:.1f}"
                    )
                try:
                    # Use "QRF+S" options: Q=quiet, R=use fit range, F=Fumili, S=return fit result
                    # hist.Fit(fit, "R+")

                    # fit_result = hist.Fit(fit, "QRS")
                    fit_result = hist.Fit(fit, "RQS")
                    fit_result = hist.Fit(fit, "RS")
                    fit_result = hist.Fit(fit, "RMS")
                    print(
                        "status =",
                        int(fit_result),
                        "cov =",
                        fit_result.CovMatrixStatus(),
                        "edm =",
                        fit_result.Edm(),
                    )

                except Exception as e:
                    print(f"Error fitting histogram {histname}: {e}")
                    fit.Delete()
                    hist.Delete()
                    continue
                # Save the fit parameters to a text file
                with open(
                    os.path.join(
                        output_dir,
                        f"m{mass}",
                        f"fit_parameters_m{mass}_{cat}_{histname}.txt",
                    ),
                    "w",
                ) as f_out:
                    f_out.write(f"Mass: {mass}\n")
                    f_out.write(f"Category: {cat}\n")
                    f_out.write(f"Histogram: {histname}\n")
                    f_out.write(
                        f"Fit type: {'Double-sided crystal ball' if use_double_crystal_ball else 'Single-sided crystal ball'}\n"
                    )
                    npar = fit.GetNpar()
                    f_out.write(
                        "Fit parameters: "
                        + ", ".join(str(fit.GetParameter(i)) for i in range(npar))
                        + "\n"
                    )
                # Save the fit plot to a filels
                c = ROOT.TCanvas(f"c_{mass}_{cat}_{histname}", "c", 800, 600)
                hist.Draw("E")
                hist.SetName(f"{hist.GetName()} m{mass}")
                fit.Draw("same")
                c.SaveAs(
                    os.path.join(
                        output_dir, "plots", f"fit_{cat}_{histname}_m{mass}.pdf"
                    )
                )
                c.Clear()

                # Take this fit, create a new histogram with the same binning as the original histogram, and fill it with the fit function values. Then save this histogram to a new root file. Include the old histogram in the new root file as well for comparison.
                fit_hist = hist.Clone()
                fit_hist.SetName("fit_hist")
                fit_hist.SetTitle("Fit histogram")
                for i in range(1, fit_hist.GetNbinsX() + 1):
                    x_low = fit_hist.GetXaxis().GetBinLowEdge(i)
                    x_high = fit_hist.GetXaxis().GetBinUpEdge(i)
                    bin_width = x_high - x_low
                    
                    # Integrate the fit function over the bin
                    # Using the Integral method of TF1
                    integral = fit.Integral(x_low, x_high)
                    
                    # For histogram, we want density * width = integral
                    # So bin content = integral / bin_width * bin_width = integral
                    fit_hist.SetBinContent(i, integral)
                    fit_hist.SetBinError(i, 0.2 * integral)  # Adjust error accordingly
                    # Set the error to 70% of the fit value for visualization purposes
                # Save the fit histogram and the original histogram to a new root file
                hist.Write(f"Original_{histname}")
                fit_hist.Write(f"fit_{histname}")

                # Draw the 2 histograms (without the fit line) on the same canvas and save it as a canvas to the root file
                hist.GetListOfFunctions().Clear()  # Remove the fit line from the fit histogram
                fit_hist.GetListOfFunctions().Clear()  # Remove the fit line from the fit histogram
                c = ROOT.TCanvas(f"c_compare_{mass}_{cat}_{histname}", "c", 800, 600)
                hist.SetLineColor(ROOT.kRed)
                hist.SetMarkerColor(ROOT.kRed)
                hist.SetMarkerStyle(20)
                hist.SetMarkerSize(0.5)
                hist.Draw("E")
                fit_hist.SetLineColor(ROOT.kBlue)
                fit_hist.SetMarkerColor(ROOT.kBlue)
                fit_hist.Draw("same")
                c.Write(f"both_{histname}")
                c.Clear()

                # Cleanup
                fit.Delete()
                fit_hist.Delete()
                hist.Delete()

            output_file.Close()

            f.Close()

    return output_filepath


def combine_shapes(filepath, masslist, catlist, output_dir, bkg_list):
    # Load all shapes in fit_outputs and combine into a single file, with new naming scheme m{mass}_{histName} for hists

    hists_to_combine = [ "signal", "background", "fit_DY", "fit_background", "data_obs" ] + bkg_list
    # hists_to_combine = [
    #     "background",
    #     "DY",
    #     "TT",
    #     "ST",
    #     "SMHiggs",
    #     "VV",
    #     "Other",
    #     "signal",
    #     "fit_DY",
    #     "fit_background",
    #     "data_obs",
    # ]

    os.makedirs(output_dir, exist_ok=True)

    output_filepath = os.path.join(output_dir, "combined_shapes_{cat}.root")

    # Loop over the mass and category, load the background distribution, and create a fit
    for cat in catlist:
        output_file = ROOT.TFile(output_filepath.format(cat=cat), "RECREATE")
        for mass in masslist:
            file_name = filepath.format(mass=mass, cat=cat)
            if not os.path.exists(file_name):
                print(f"File {file_name} does not exist, skipping.")
                continue
            f = ROOT.TFile(file_name)

            for histname in hists_to_combine:
                hist = f.Get(histname)
                if hist is None:
                    print(
                        f"Histogram {histname} not found in file {file_name}, skipping."
                    )
                    continue

                # Clone the histogram to keep it valid after closing the file
                print(f"Cloning {histname} from file {file_name}")
                hist = hist.Clone()
                hist.SetName(f"m{mass}_{histname}")
                if histname == "data_obs":
                    hist.SetName("data_obs")
                output_file.cd()  # Switch to output file before writing
                hist.Write()
                hist.Delete()

            f.Close()
        output_file.Close()
    return output_filepath


def rebin_shapes(
    filepath,
    masslist,
    catlist,
    output_dir,
    bkgs_to_consider_resolved,
    bkgs_to_consider_boosted,
    bkg_list,
    nTotalBins=None,
):
    import array

    os.makedirs(output_dir, exist_ok=True)

    # histograms to carry through into rebinned output
    rebin_hist_names = [ "signal", "background", "fit_DY", "fit_background" ] + bkg_list
    # rebin_hist_names = [
    #     "background",
    #     "DY",
    #     "TT",
    #     "ST",
    #     "SMHiggs",
    #     "VV",
    #     "Other",
    #     "signal",
    #     "fit_DY",
    #     "fit_background",
    # ]

    output_filepath = os.path.join(output_dir, "rebin_combined_shapes_{cat}.root")

    def integral_in_range(hist, lo_bin, hi_bin):
        return float(hist.Integral(lo_bin, hi_bin))

    def build_rebin_edges(hist_dict, mass, cat, nTotalBins=None):
        """
        Build variable-width bin edges from right to left.

        Requirement:
          - each final bin must have strictly positive content in every individual background

        Greedy optimization:
          - for the current right edge, scan leftward candidate starts
          - among valid candidates choose the one maximizing S/sqrt(S+B)
        """
        signal_hist = hist_dict["signal"]
        nbins = signal_hist.GetNbinsX()
        xaxis = signal_hist.GetXaxis()

        # Always keep the upper edge of the last bin
        edges = [xaxis.GetBinUpEdge(nbins)]
        scores = []

        current_right_bin = nbins

        total_signal = integral_in_range(signal_hist, 0, nbins)
        if nTotalBins:
            signal_per_bin = total_signal / nTotalBins

        while current_right_bin >= 1:
            best_left_bin = None
            best_score = 0.0

            # scan candidate merged bins [left_bin, current_right_bin]
            for left_bin in range(current_right_bin, 0, -1):
                # require each background > 0 in the merged bin
                valid = True
                total_b = 0.0
                for bkg in bkg_names:
                    bval = integral_in_range(
                        hist_dict[bkg], left_bin, current_right_bin
                    )
                    if bval <= 0.0:
                        valid = False
                        break
                    total_b += bval

                if not valid:
                    continue

                sval = integral_in_range(signal_hist, left_bin, current_right_bin)

                if nTotalBins and sval < signal_per_bin:
                    continue

                denom = sval + total_b
                # denom = total_b
                score = 0.0
                if denom > 0.0:
                    score = sval / math.sqrt(denom)

                # choose the candidate with largest local significance
                if score > best_score:
                    best_score = score
                    best_left_bin = left_bin
                    break

            # fallback: if no valid bin satisfies all bkgs > 0,
            # merge everything remaining into one last bin
            if best_left_bin is None:
                best_left_bin = 1

            # append lower edge of chosen bin
            low_edge = xaxis.GetBinLowEdge(best_left_bin)
            edges.append(low_edge)
            scores.append(best_score)

            # continue to the left of the chosen bin
            current_right_bin = best_left_bin - 1

        # edges were built right->left, convert to increasing order
        edges = sorted(set(edges))

        print(f"Finished significances (right to left) {scores}")

        # safety: ensure at least 2 edges
        if len(edges) < 2:
            edges = [xaxis.GetXmin(), xaxis.GetXmax()]

        return edges

    for cat in catlist:
        bkg_names = (
            bkgs_to_consider_boosted if cat == "boosted" else bkgs_to_consider_resolved
        )
        in_name = filepath.format(cat=cat)
        if not os.path.exists(in_name):
            print(f"Input file {in_name} does not exist, skipping category {cat}")
            continue

        input_file = ROOT.TFile(in_name, "READ")
        output_file = ROOT.TFile(output_filepath.format(cat=cat), "RECREATE")

        # preserve data_obs if present
        data_obs = input_file.Get("data_obs")
        if data_obs:
            data_obs = data_obs.Clone("data_obs")
            data_obs.SetDirectory(0)

        for mass in masslist:
            print(f"Rebinning mass {mass}, category {cat}")
            hist_dict = {}

            # load the mass-specific histograms
            for histname in rebin_hist_names:
                hname = f"m{mass}_{histname}"
                hist = input_file.Get(hname)
                if hist is None:
                    print(
                        f"  Histogram {hname} not found in {input_file.GetName()}, skipping."
                    )
                    continue
                hist = hist.Clone(hname)
                hist.SetDirectory(0)
                hist_dict[histname] = hist

            # require signal and all separate backgrounds
            required = ["signal", "DY", "TT", "Other", "background"]
            if any(name not in hist_dict for name in required):
                print(
                    f"  Missing required histograms for mass {mass}, category {cat}, skipping."
                )
                for h in hist_dict.values():
                    h.Delete()
                continue

            # build optimized variable binning
            edges = build_rebin_edges(hist_dict, mass, cat, nTotalBins)
            print(f"  New bin edges for m{mass}, {cat}: {edges}")

            edge_array = array.array("d", edges)
            nbins_new = len(edges) - 1

            output_file.cd()

            # rebin all available mass histograms
            rebinned = {}
            for histname, hist in hist_dict.items():
                new_name = f"m{mass}_{histname}"
                rebinned_hist = hist.Rebin(nbins_new, new_name, edge_array)
                rebinned_hist.SetDirectory(output_file)
                rebinned[new_name] = rebinned_hist

            # also create rebinned data_obs for this mass if desired
            # Since combine usually expects one data_obs per category, not per mass,
            # we do NOT overwrite the global "data_obs" object here.
            # If you want a mass-specific copy for debugging, uncomment:
            #
            # if data_obs:
            #     data_obs_mass = data_obs.Rebin(nbins_new, f"m{mass}_data_obs", edge_array)
            #     data_obs_mass.SetDirectory(output_file)

            # write rebinned hists
            for name, rh in rebinned.items():
                rh.Write()

            # optional: save a simple canvas for signal/background after rebinning
            if f"m{mass}_signal" in rebinned and f"m{mass}_background" in rebinned:
                c = ROOT.TCanvas(
                    f"c_rebin_m{mass}_{cat}", f"rebin m{mass} {cat}", 800, 600
                )
                rebinned[f"m{mass}_signal"].SetLineColor(ROOT.kRed)
                rebinned[f"m{mass}_signal"].SetLineWidth(2)
                rebinned[f"m{mass}_background"].SetLineColor(ROOT.kBlue)
                rebinned[f"m{mass}_background"].SetLineWidth(2)
                rebinned[f"m{mass}_background"].Draw("HIST")
                rebinned[f"m{mass}_signal"].Draw("HIST SAME")
                c.BuildLegend()
                c.Write(f"c_rebin_m{mass}_{cat}")
                c.Close()

            # cleanup clones loaded from input
            for h in hist_dict.values():
                h.Delete()

        # optionally copy original category-level data_obs through
        if data_obs:
            output_file.cd()
            data_obs.Write("data_obs")
            data_obs.Delete()

        output_file.Close()
        input_file.Close()
    return output_filepath


def calculate_limits(filepath, masslist, catlist, output_dir):

    def limit_to_json(input_root, output_json):
        t = uproot.open(f"{input_root}:limit", branches=["limit"])
        arrays = t.arrays()
        limits = {
            "exp-2": arrays.limit[0],
            "exp-1": arrays.limit[1],
            "exp": arrays.limit[2],
            "exp+1": arrays.limit[3],
            "exp+2": arrays.limit[4],
            "observed": arrays.limit[5],
        }
        with open(output_json, "w") as fp:
            json.dump(limits, fp)

        os.system(f"rm {input_root}")
        os.system(f"rm combine_logger.out")

    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(
        os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Studies",
            "ModelValidation",
            "config",
            "call_combine_benchmark.sh",
        ),
        os.path.join(output_dir, "."),
    )
    shutil.copy(
        os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Studies",
            "ModelValidation",
            "config",
            "Run3card_boosted_local.txt",
        ),
        os.path.join(output_dir, "."),
    )
    shutil.copy(
        os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Studies",
            "ModelValidation",
            "config",
            "Run3card_res2b_local.txt",
        ),
        os.path.join(output_dir, "."),
    )
    shutil.copy(
        os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Studies",
            "ModelValidation",
            "config",
            "Run3card_recovery_local.txt",
        ),
        os.path.join(output_dir, "."),
    )
    shutil.copy(
        os.path.join(
            os.environ["ANALYSIS_PATH"],
            "Studies",
            "ModelValidation",
            "config",
            "Run3card_combined_local.txt",
        ),
        os.path.join(output_dir, "."),
    )

    if "res2b" in catlist:
        shutil.copy(
            filepath.format(cat="res2b"), os.path.join(output_dir, "run3_res2b.root")
        )
    if "res1b" in catlist:
        shutil.copy(
            filepath.format(cat="res1b"), os.path.join(output_dir, "run3_res1b.root")
        )
    if "boosted" in catlist:
        shutil.copy(
            filepath.format(cat="boosted"),
            os.path.join(output_dir, "run3_boosted.root"),
        )

    for mass in masslist:
        if "res2b" in catlist:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    os.path.join(output_dir, "Run3card_res2b_local.txt"),
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_res2b",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )

            limit_to_json(
                f"higgsCombine_m{mass}_res2b.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_dir, f"m{mass}_res2b.json"),
            )

        if "res1b" in catlist:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    os.path.join(output_dir, "Run3card_recovery_local.txt"),
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_res1b",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )

            limit_to_json(
                f"higgsCombine_m{mass}_res1b.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_dir, f"m{mass}_res1b.json"),
            )

        if "boosted" in catlist:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    os.path.join(output_dir, "Run3card_boosted_local.txt"),
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_boosted",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )

            limit_to_json(
                f"higgsCombine_m{mass}_boosted.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_dir, f"m{mass}_boosted.json"),
            )

        if "res2b" in catlist and "res1b" in catlist and "boosted" in catlist:
            ps_call(
                [
                    "combine",
                    "-M",
                    "AsymptoticLimits",
                    os.path.join(output_dir, "Run3card_combined_local.txt"),
                    "--rMax",
                    "1",
                    "-t",
                    "-1",
                    "-n",
                    f"_m{mass}_combined",
                    "-m",
                    f"{mass}",
                ],
                env=cmssw_env,
            )

            limit_to_json(
                f"higgsCombine_m{mass}_combined.AsymptoticLimits.mH{mass}.root",
                os.path.join(output_dir, f"m{mass}_combined.json"),
            )

    return


def plot_limits_from_json(json_dir, masslist, catlist, output_dir, draw_observed=True):
    """
    Read limit JSON files of the form:
        m{mass}_{cat}.json

    Each JSON contains:
        {
            "exp-2": ...,
            "exp-1": ...,
            "exp": ...,
            "exp+1": ...,
            "exp+2": ...,
            "observed": ...
        }

    Produce:
      - one standard Brazil-band limit plot per category
      - one comparison plot of expected limits for all categories
    """

    run2_2018 = {
        "res2b": {
            "mass": [300, 400, 500, 550, 600, 650, 700, 800, 900],
            "exp": [
                5.4275,
                2.0375,
                0.4641,
                0.3125,
                0.2570,
                0.2289,
                0.2266,
                0.2242,
                0.2773,
            ],
        },
        "res1b": {
            "mass": [300, 400, 500, 550, 600, 650, 700, 800, 900],
            "exp": [
                9.6375,
                3.85,
                1.4062,
                1.2313,
                0.8750,
                0.7906,
                0.7344,
                0.6438,
                0.5781,
            ],
        },
        "boosted": {
            "mass": [300, 400, 500, 550, 600, 650, 700, 800, 900],
            "exp": [62.7, 9.975, 2.125, 1.2969, 0.6844, 0.3422, 0.193, 0.1152, 0.0785],
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    limit_data = {}

    if "res2b" in catlist and "res1b" in catlist and "boosted" in catlist:
        catlist.append("combined")

    # Collect all category/mass values
    for cat in catlist:
        limit_data[cat] = {
            "mass": [],
            "exp-2": [],
            "exp-1": [],
            "exp": [],
            "exp+1": [],
            "exp+2": [],
            "observed": [],
        }

        for mass in masslist:
            json_path = os.path.join(json_dir, f"m{mass}_{cat}.json")
            if not os.path.exists(json_path):
                print(f"Missing JSON file: {json_path}, skipping")
                continue

            with open(json_path, "r") as fp:
                vals = json.load(fp)

            limit_data[cat]["mass"].append(mass)
            limit_data[cat]["exp-2"].append(vals["exp-2"])
            limit_data[cat]["exp-1"].append(vals["exp-1"])
            limit_data[cat]["exp"].append(vals["exp"])
            limit_data[cat]["exp+1"].append(vals["exp+1"])
            limit_data[cat]["exp+2"].append(vals["exp+2"])
            limit_data[cat]["observed"].append(vals["observed"])

    # Make one Brazil plot per category
    y_min = 0.001
    y_max = 100
    x_min = 200
    x_max = 1000
    for cat in catlist:
        data = limit_data[cat]
        if len(data["mass"]) == 0:
            print(f"No data found for category {cat}, skipping plot")
            continue

        masses = data["mass"]

        plt.figure(figsize=(8, 6))

        # 2 sigma band (yellow)
        plt.fill_between(
            masses,
            data["exp-2"],
            data["exp+2"],
            color="gold",
            label=r"Expected $\pm 2\sigma$",
        )

        # 1 sigma band (green)
        plt.fill_between(
            masses,
            data["exp-1"],
            data["exp+1"],
            color="limegreen",
            label=r"Expected $\pm 1\sigma$",
        )

        # Expected median
        plt.plot(
            masses,
            data["exp"],
            linestyle="--",
            color="black",
            linewidth=2,
            label="Expected",
        )

        # Observed
        if draw_observed:
            plt.plot(
                masses,
                data["observed"],
                linestyle="-",
                color="black",
                linewidth=2,
                label="Observed",
            )

        if cat != "combined":
            plt.plot(
                run2_2018[cat]["mass"],
                run2_2018[cat]["exp"],
                linestyle=":",
                color="blue",
                linewidth=2,
                label=f"2018 {cat}",
            )

        plt.xlabel("Mass [GeV]")
        plt.ylabel("95% CL limit on signal strength")
        plt.title(f"Limits: {cat}")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.yscale("log")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(output_dir, f"limits_{cat}.png")
        out_pdf = os.path.join(output_dir, f"limits_{cat}.pdf")
        # plt.savefig(out_png)
        plt.savefig(out_pdf)
        plt.close()

    # Make one comparison plot of expected limits for all categories
    plt.figure(figsize=(8, 6))
    color_map = {
        "res2b": "tab:blue",
        "res1b": "tab:red",
        "boosted": "tab:green",
        "combined": "tab:purple",
    }

    has_any = False
    for cat in catlist:
        data = limit_data[cat]
        if len(data["mass"]) == 0:
            continue
        has_any = True
        plt.plot(
            data["mass"],
            data["exp"],
            linestyle="--",
            linewidth=2,
            color=color_map.get(cat, None),
            label=f"{cat} expected",
        )
        if draw_observed:
            plt.plot(
                data["mass"],
                data["observed"],
                linestyle="-",
                linewidth=2,
                color=color_map.get(cat, None),
                alpha=0.8,
                label=f"{cat} observed",
            )

    if has_any:
        plt.xlabel("Mass [GeV]")
        plt.ylabel("95% CL limit on signal strength")
        plt.title("Limit comparison")
        plt.grid(True, linestyle=":", alpha=0.5)
        plt.yscale("log")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(output_dir, "limits_comparison.png")
        out_pdf = os.path.join(output_dir, "limits_comparison.pdf")
        # plt.savefig(out_png)
        plt.savefig(out_pdf)
    plt.close()


def prepare_shapes():
    # training_dir_resolved = "/eos/user/d/daebi/HH_bbWW/DNNTraining/25Apr_Resolved_v1_Logits/Run3_2022EE/DNN_DoubleLepton_Resolved_Training0_par{par}_m{mass}"
    training_dir_resolved = "/eos/user/d/daebi/HH_bbWW/DNNTraining/23May_Resolved_v1/Run3_2022EE/DNN_DoubleLepton_Resolved_Training0_par{par}_m{mass}"
    training_dir_boosted = "/eos/user/d/daebi/HH_bbWW/DNNTraining/9May_Boosted_v1/Run3_2022EE/DNN_DoubleLepton_Boosted_Training0_par{par}_m{mass}"

    # training_dir_resolved = "/eos/user/d/daebi/HH_bbWW/DNNTraining/training_v4/Run3_2022EE/DNN_DoubleLepton_Resolved_Training0_par{par}_m0" # Parametric, all named m0

    # catlist = ["res2b", "res1b", "boosted"]
    catlist = ["res2b", "res1b" ]

    # all_bkgs = ["TT", "DY", "ST", "SMHiggs", "VV", "Other"]
    # all_bkgs = ["TT", "DY", "ST", "Other"]
    all_bkgs = ["TT", "DY", "Other"]

    # bkgs_to_consider_resolved = ["DY", "TT", "ST", "Other"]
    # bkgs_to_consider_resolved = ["fit_DY", "TT"]
    bkgs_to_consider_resolved = ["DY", "TT"]
    bkgs_to_consider_boosted = ["fit_background"]

    # output_dir = "LocalLimits/25Apr_Resolved_9May_Boosted_10Bins_noHME"
    output_dir = "LocalLimits/23May_Resolved_10Bins_NoFit_noHME"

    # Step 1: hadd the separate parity files per mass point
    masslist = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]

    filepath_resolved = os.path.join(
        # training_dir_resolved, "validation", "validation_logit_{cat}_hme_cut.root"
        training_dir_resolved, "validation", "validation_logit_{cat}.root"
    )
    filepath_boosted = os.path.join(
        # training_dir_boosted, "validation", "validation_logit_{cat}_hme_cut.root"
        training_dir_boosted, "validation", "validation_logit_{cat}.root"
    )
    histname = "m{mass}_{proc}_class0"

    output_dir_hadd = os.path.join(output_dir, "hadd_outputs")

    hadded_filepath = hadd_parity_files(
        filepath_resolved,
        filepath_boosted,
        histname,
        masslist,
        catlist,
        output_dir_hadd,
        all_bkgs,
    )

    # Step 2: fit the hadded backgrounds and sum(bkg)
    output_dir_fit = os.path.join(output_dir, "fit_outputs")
    fitted_filepath = fit_hadded_shapes(
        hadded_filepath, masslist, catlist, output_dir_fit, all_bkgs
    )

    # Step 3: merge into one shape file with correct naming scheme
    output_dir_combined = os.path.join(output_dir, "combined_shapes")
    combined_filepath = combine_shapes(
        fitted_filepath, masslist, catlist, output_dir_combined, all_bkgs
    )

    # Step 4: rebin the combined shapes for combine
    output_dir_rebin = os.path.join(output_dir, "rebin_combined_shapes")
    rebin_filepath = rebin_shapes(
        combined_filepath,
        masslist,
        catlist,
        output_dir_rebin,
        bkgs_to_consider_resolved,
        bkgs_to_consider_boosted,
        all_bkgs,
        10,  # nTotalBins, or none, or comment out
    )

    # Step 5: calculate limits of new shapes
    output_dir_limits = os.path.join(output_dir, "limits")
    calculate_limits(rebin_filepath, masslist, catlist, output_dir_limits)

    # Step 6: plot the limits for comparison
    output_dir_plots = os.path.join(output_dir, "plots")
    plot_limits_from_json(output_dir_limits, masslist, catlist, output_dir_plots)


if __name__ == "__main__":
    prepare_shapes()
