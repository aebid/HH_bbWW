import uproot
import numpy as np
import os


def main():
    # year = 2018
    # year = "run3"
    year = "run3_new"

    if year == "run3":
        masspoints = [300, 400, 500, 550, 600, 650, 700, 800, 900]

        backgrounds = ["m{mp}_TT_class0", "m{mp}_DY_class0", "m{mp}_Other_class0"]
        print_bkgs = ["TT", "DY", "Other"]

        signals = ["m{mp}_Signal_class0"]

        categories = ["res2b", "res1b", "boosted"]

        file_name_format = os.path.join(
            "{year}_shapes", "{mp}", "run3_{category}_m{mp}.root"
        )

    elif year == "run3_new":
        masspoints = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]

        backgrounds = ["m{mp}_TT", "m{mp}_DY", "m{mp}_Other", "m{mp}_fit_DY"]
        print_bkgs = ["TT", "DY", "Other", "fit_DY"]

        signals = ["m{mp}_signal"]

        categories = ["res2b", "res1b", "boosted"]

        file_name_format = os.path.join(
            "{year}_shapes", "rebin_combined_shapes_{category}.root"
        )

    elif year == "run3_parametric":
        masspoints = [300, 400, 500, 550, 600, 650, 700, 800, 900, 1000]

        backgrounds = ["m{mp}_TT", "m{mp}_DY", "m{mp}_Other", "m{mp}_fit_DY"]
        print_bkgs = ["TT", "DY", "Other", "fit_DY"]

        signals = ["m{mp}_signal"]

        categories = ["res2b", "res1b", "boosted"]

        file_name_format = os.path.join(
            "{year}_shapes", "rebin_combined_shapes_{category}.root"
        )

    else:
        masspoints = [300, 400, 500, 550, 600, 650, 700, 800, 900]
        backgrounds = [
            "TT",
            "DY",
            "ST",
            "Fakes",
            "ttW",
            "Other_bbWW",
            "ttZ",
            "VH_hww",
            "tHW_hww",
            "tHq_hww",
            "ZH_hww",
            "ZH_htt",
            "ZH_hbb",
            "WH_hbb",
            "ttH_hww",
            "ttH_hbb",
            "qqH_hzz",
            "qqH_hww",
            "qqH_htt",
            "qqH_hmm",
            "qqH_hgg",
            "qqH_hbb",
            "ggH_hzz",
            "ggH_hww",
            "ggH_htt",
            "ggH_hmm",
            "ggH_hgg",
            "ggH_hbb",
            "VVV",
            "VV",
            "WJets",
        ]
        print_bkgs = ["TT", "DY", "ST", "Fakes"]

        # signals = [ "signal_ggf_spin0_{mp}_hbbhtt", "signal_ggf_spin0_{mp}_hbbhww" ]
        signals = ["signal_ggf_spin0_{mp}_hbbhww"]

        categories = ["resolved2b", "resolved1b", "boosted"]

        file_name_format = os.path.join(
            "{year}_shapes", "{mp}", "HH_DL_{mp}_{category}_GGF_{year}.root"
        )

    # percentage
    signif_inc_thresh = 1.03

    for mp in masspoints:
        print(f"Processing mass point {mp} GeV")

        source = np.empty(0)
        sig_counts = np.empty(0)
        sig_errors = np.empty(0)
        total_bkg_counts = np.empty(0)
        bkg_counts_dict = {}
        bkg_errors_dict = {}
        for bkg in backgrounds:
            bkg = bkg.format(mp=mp)
            bkg_counts_dict[bkg] = np.empty(0)
            bkg_errors_dict[bkg] = np.empty(0)

        # Loop all categories in catlist and concatenate them into single arrays for signal and each background
        for category in categories:

            file_name = file_name_format.format(year=year, mp=mp, category=category)

            f = uproot.open(file_name)

            temp_sig_counts = None
            temp_sig_errors = None
            for sig in signals:
                sig = sig.format(mp=mp)
                sig_hist = f[sig]

                this_sig_counts, _ = sig_hist.to_numpy()
                this_sig_errors = np.sqrt(sig_hist.variances())

                if temp_sig_counts is None:
                    temp_sig_counts = np.zeros_like(this_sig_counts)
                    temp_sig_errors = np.zeros_like(this_sig_errors)

                temp_sig_counts += this_sig_counts
                temp_sig_errors += np.power(this_sig_errors, 2)
            temp_sig_errors = np.sqrt(temp_sig_errors)

            sig_counts = np.concatenate((sig_counts, temp_sig_counts))
            sig_errors = np.concatenate((sig_errors, temp_sig_errors))

            this_source = np.array([f"{category}"] * len(this_sig_counts))
            source = np.concatenate((source, this_source))

            this_total_bkg_counts = 0
            for bkg in backgrounds:
                bkg = bkg.format(mp=mp)

                bkg_hist = f[bkg]
                bkg_counts, _ = bkg_hist.to_numpy()
                bkg_errors = np.sqrt(bkg_hist.variances())
                bkg_counts_dict[bkg] = np.concatenate(
                    (bkg_counts_dict[bkg], bkg_counts)
                )
                bkg_errors_dict[bkg] = np.concatenate(
                    (bkg_errors_dict[bkg], bkg_errors)
                )
                if bkg == "m{mp}_DY".format(mp=mp):
                    continue
                this_total_bkg_counts += bkg_counts
            total_bkg_counts = np.concatenate((total_bkg_counts, this_total_bkg_counts))

        n_bins = len(sig_counts)

        print(
            f"Full shape has nSignal: {np.sum(sig_counts)} -- and nBackground: {np.sum(total_bkg_counts)}"
        )

        bin_significances = sig_counts / np.sqrt(total_bkg_counts)
        # bin_significances = sig_counts/np.sqrt(total_bkg_counts + sig_counts)
        # bin_significances = sig_counts/total_bkg_counts

        # Sort the bins by significance
        sorted_bin_indices = np.argsort(bin_significances)[::-1]
        # print(f"Bins by significance: {sorted_bin_indices}")
        significance_sorted = bin_significances[sorted_bin_indices]
        source_sorted = source[sorted_bin_indices]
        sig_sorted = sig_counts[sorted_bin_indices]
        sig_error_sorted = sig_errors[sorted_bin_indices]
        bkg_sorted = total_bkg_counts[sorted_bin_indices]
        bkg_error_sorted = np.sqrt(
            np.sum(
                [
                    np.power(bkg_errors_dict[bkg.format(mp=mp)][sorted_bin_indices], 2)
                    for bkg in backgrounds
                ],
                axis=0,
            )
        )
        bkg_sorted_dict = {}
        bkg_error_sorted_dict = {}
        for bkg in backgrounds:
            bkg = bkg.format(mp=mp)
            bkg_sorted_dict[bkg] = bkg_counts_dict[bkg][sorted_bin_indices]
            bkg_error_sorted_dict[bkg] = bkg_errors_dict[bkg][sorted_bin_indices]

        sig_sorted_integrated = np.zeros(n_bins)
        sig_error_sorted_integrated = np.zeros(n_bins)
        bkg_sorted_integrated = np.zeros(n_bins)
        bkg_error_sorted_integrated = np.zeros(n_bins)
        bkg_sorted_integrated_dict = {}
        bkg_error_sorted_integrated_dict = {}
        for bkg in backgrounds:
            bkg = bkg.format(mp=mp)
            bkg_sorted_integrated_dict[bkg] = np.zeros(n_bins)
            bkg_error_sorted_integrated_dict[bkg] = np.zeros(n_bins)
        signif_sorted_merged_integrated = np.zeros(n_bins)
        signif_sorted_integrated = np.zeros(n_bins)

        # Integrate each process per bin and check significance
        for bin_num in range(n_bins):
            sig_sorted_integrated[bin_num] = np.sum(sig_sorted[: bin_num + 1])
            sig_error_sorted_integrated[bin_num] = np.sqrt(
                np.sum(np.power(sig_error_sorted[: bin_num + 1], 2))
            )
            bkg_sorted_integrated[bin_num] = np.sum(bkg_sorted[: bin_num + 1])
            bkg_error_sorted_integrated[bin_num] = np.sqrt(
                np.sum(np.power(bkg_error_sorted[: bin_num + 1], 2))
            )
            for bkg in backgrounds:
                bkg = bkg.format(mp=mp)
                bkg_sorted_integrated_dict[bkg][bin_num] = np.sum(
                    bkg_sorted_dict[bkg][: bin_num + 1]
                )
                bkg_error_sorted_integrated_dict[bkg][bin_num] = np.sqrt(
                    np.sum(np.power(bkg_error_sorted_dict[bkg][: bin_num + 1], 2))
                )
            signif_sorted_merged_integrated[bin_num] = sig_sorted_integrated[
                bin_num
            ] / np.sqrt(bkg_sorted_integrated[bin_num])

            # Combine significances by sqrt(sum of squares) instead of sum of counts
            signif_sorted_integrated[bin_num] = np.sqrt(
                np.sum(np.power(significance_sorted[: bin_num + 1], 2))
            )

        print(f"Significance merged: {signif_sorted_merged_integrated}")

        best_sig_idx = 0
        for bin_num in range(n_bins - 1):
            if (
                signif_sorted_merged_integrated[bin_num + 1]
                / signif_sorted_merged_integrated[bin_num]
                > signif_inc_thresh
            ):
                best_sig_idx = bin_num + 1
            else:
                break

        print(
            f"Found the bin with best significance {best_sig_idx}, showing all sorted distributions"
        )

        print(f"Integrated Signal: {sig_sorted_integrated}")
        print(f"Signal errors integrated: {sig_error_sorted_integrated}")
        print(f"Integrated Background: {bkg_sorted_integrated}")
        print(f"Background errors integrated: {bkg_error_sorted_integrated}")
        print(
            f"Error % on background: {100 * bkg_error_sorted_integrated / bkg_sorted_integrated}"
        )
        for bkg, bkg_counts in bkg_sorted_integrated_dict.items():
            # if bkg not in print_bkgs:
            #     continue
            print(f"{bkg} integrated: {bkg_counts}")
            print(f"{bkg} errors integrated: {bkg_error_sorted_integrated_dict[bkg]}")
        print(f"Integrated Significance: {signif_sorted_merged_integrated}")
        print(f"Source: {source_sorted}")


if __name__ == "__main__":
    main()
