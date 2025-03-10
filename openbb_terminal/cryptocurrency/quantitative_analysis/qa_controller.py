"""Quantitative Analysis Controller Module"""
__docformat__ = "numpy"

import argparse
import logging
from typing import List

import numpy as np
import pandas as pd

from openbb_terminal.custom_prompt_toolkit import NestedCompleter

from openbb_terminal import feature_flags as obbff
from openbb_terminal.common.quantitative_analysis import qa_view, rolling_view
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    EXPORT_ONLY_FIGURES_ALLOWED,
    EXPORT_ONLY_RAW_DATA_ALLOWED,
    check_positive,
    check_proportion_range,
    check_list_dates,
)
from openbb_terminal.menu import session
from openbb_terminal.parent_classes import CryptoBaseController
from openbb_terminal.rich_config import console, MenuText

logger = logging.getLogger(__name__)


class QaController(CryptoBaseController):
    """Quantitative Analysis Controller class"""

    CHOICES_COMMANDS = [
        "load",
        "pick",
        "raw",
        "summary",
        "line",
        "hist",
        "cdf",
        "bw",
        "rolling",
        "decompose",
        "cusum",
        "acf",
        "spread",
        "quantile",
        "skew",
        "kurtosis",
        "normality",
        "qqplot",
        "unitroot",
        "goodness",
        "unitroot",
    ]
    FULLER_REG = ["c", "ct", "ctt", "nc"]
    KPS_REG = ["c", "ct"]

    PATH = "/crypto/qa/"

    def __init__(
        self,
        symbol: str,
        data: pd.DataFrame,
        queue: List[str] = None,
    ):
        """Constructor"""
        super().__init__(queue)

        data["Returns"] = data["Close"].pct_change()
        data["LogRet"] = np.log(data["Close"]) - np.log(data["Close"].shift(1))
        data = data.dropna()

        self.data = data
        self.symbol = symbol
        self.target = "Close"

        if session and obbff.USE_PROMPT_TOOLKIT:
            choices: dict = {c: {} for c in self.controller_choices}
            choices["pick"] = {c: {} for c in list(data.columns)}
            choices["load"] = {
                "--interval": {
                    c: {}
                    for c in [
                        "1",
                        "5",
                        "15",
                        "30",
                        "60",
                        "240",
                        "1440",
                        "10080",
                        "43200",
                    ]
                },
                "-i": "--interval",
                "--exchange": {c: {} for c in self.exchanges},
                "--source": {c: {} for c in ["CCXT", "YahooFinance", "CoingGecko"]},
                "--vs": {c: {} for c in ["usd", "eur"]},
                "--start": None,
                "-s": "--start",
                "--end": None,
                "-e": "--end",
            }
            choices["unitroot"] = {
                "--fuller_reg": {c: {} for c in self.FULLER_REG},
                "-r": "--fuller_reg",
                "--kps_reg": {c: {} for c in self.KPS_REG},
                "-k": "--kps_reg",
            }
            choices["line"] = {
                "--log": {},
                "--draw": {},
                "-d": "--draw",
                "--ml": None,
                "--ms": None,
            }
            choices["hist"] = {
                "--bins": {str(c): {} for c in range(10, 100)},
                "-b": "--bins",
            }
            choices["bw"] = {
                "--yearly": {},
                "-y": {},
            }
            choices["acf"] = {
                "--lags": {str(c): {} for c in range(5, 100)},
                "-l": "--lags",
            }
            choices["rolling"] = {
                "--window": {str(c): {} for c in range(5, 100)},
                "-w": "--window",
            }
            choices["spread"] = {
                "--window": {str(c): {} for c in range(5, 100)},
                "-w": "--window",
            }
            choices["quantile"] = {
                "--window": {str(c): {} for c in range(5, 100)},
                "-w": "--window",
                "--quantile": {str(c): {} for c in np.arange(0.0, 1.0, 0.01)},
                "-q": "--quantile",
            }
            choices["skew"] = {
                "--window": {str(c): {} for c in range(5, 100)},
                "-w": "--window",
            }
            choices["kurtosis"] = {
                "--window": {str(c): {} for c in range(5, 100)},
                "-w": "--window",
            }
            choices["raw"] = {
                "--limit": {str(c): {} for c in range(1, 100)},
                "-l": "--limit",
                "--descend": {},
            }
            choices["decompose"] = {
                "--multiplicative": None,
                "-m": "--multiplicative",
            }
            choices["cusum"] = {
                "--threshold": None,
                "-t": "--threshold",
                "--drift": None,
                "-d": "--drift",
            }

            choices["support"] = self.SUPPORT_CHOICES
            choices["about"] = self.ABOUT_CHOICES

            self.completer = NestedCompleter.from_nested_dict(choices)

    def print_help(self):
        """Print help"""
        mt = MenuText("crypto/qa/")
        mt.add_cmd("load")
        mt.add_cmd("pick")
        mt.add_raw("\n")
        mt.add_param("_ticker", self.symbol)
        mt.add_param("_target", self.target)
        mt.add_raw("\n")
        mt.add_info("_statistics_")
        mt.add_cmd("summary")
        mt.add_cmd("normality")
        mt.add_cmd("unitroot")
        mt.add_info("_plots_")
        mt.add_cmd("line")
        mt.add_cmd("hist")
        mt.add_cmd("cdf")
        mt.add_cmd("bw")
        mt.add_cmd("acf")
        mt.add_cmd("qqplot")
        mt.add_info("_rolling_metrics_")
        mt.add_cmd("rolling")
        mt.add_cmd("spread")
        mt.add_cmd("quantile")
        mt.add_cmd("skew")
        mt.add_cmd("kurtosis")
        mt.add_info("_other_")
        mt.add_cmd("raw")
        mt.add_cmd("decompose")
        mt.add_cmd("cusum")
        console.print(text=mt.menu_text, menu="Cryptocurrency - Quantitative Analysis")

    def custom_reset(self):
        """Class specific component of reset command"""
        if self.symbol:
            return ["crypto", f"load {self.symbol}", "qa"]
        return []

    @log_start_end(log=logger)
    def call_pick(self, other_args: List[str]):
        """Process pick command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="pick",
            description="""
                Change target variable
            """,
        )
        parser.add_argument(
            "-t",
            "--target",
            dest="target",
            choices=list(self.data.columns),
            help="Select variable to analyze",
        )
        if other_args and "-t" not in other_args and "-h" not in other_args:
            other_args.insert(0, "-t")

        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            self.target = ns_parser.target
            console.print("")

    @log_start_end(log=logger)
    def call_raw(self, other_args: List[str]):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="raw",
            description="""
                Print raw data to console
            """,
        )
        parser.add_argument(
            "-l",
            "--limit",
            help="Number to show",
            type=check_positive,
            default=20,
            dest="limit",
        )
        parser.add_argument(
            "-d",
            "--descend",
            action="store_true",
            default=False,
            dest="descend",
            help="Sort in descending order",
        )

        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_raw(
                data=self.data[self.target],
                limit=ns_parser.limit,
                sortby="Close",
                descend=ns_parser.descend,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_summary(self, other_args: List[str]):
        """Process summary command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="summary",
            description="""
                Summary statistics
            """,
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_summary(data=self.data, export=ns_parser.export)

    @log_start_end(log=logger)
    def call_line(self, other_args: List[str]):
        """Process line command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="line",
            description="Show line plot of selected data and allow to draw lines or highlight specific datetimes.",
        )
        parser.add_argument(
            "--log",
            help="Plot with y on log scale",
            dest="log",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-d",
            "--draw",
            help="Draw lines and annotate on the plot",
            dest="draw",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--ml",
            help="Draw vertical line markers to highlight certain events (comma separated dates, e.g. 2020-01-01,2020-02-01)",  # noqa: E501
            dest="ml",
            type=check_list_dates,
            default="",
        )
        parser.add_argument(
            "--ms",
            help="Draw scatter markers to highlight certain events (comma separated dates, e.g. 2021-01-01,2021-02-01)",
            dest="ms",
            type=check_list_dates,
            default="",
        )

        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_FIGURES_ALLOWED
        )
        if ns_parser:
            qa_view.display_line(
                self.data[self.target],
                title=f"{self.symbol} {self.target}",
                log_y=ns_parser.log,
                draw=ns_parser.draw,
                markers_lines=ns_parser.ml,
                markers_scatter=ns_parser.ms,
            )

    @log_start_end(log=logger)
    def call_hist(self, other_args: List[str]):
        """Process hist command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="hist",
            description="""
                Histogram with density and rug
            """,
        )
        parser.add_argument(
            "-b", "--bins", type=check_positive, default=15, dest="n_bins"
        )
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            qa_view.display_hist(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                bins=ns_parser.n_bins,
            )

    @log_start_end(log=logger)
    def call_cdf(self, other_args: List[str]):
        """Process cdf command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="cdf",
            description="""
                Cumulative distribution function
            """,
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_cdf(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_bw(self, other_args: List[str]):
        """Process bwy command"""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            add_help=False,
            prog="bw",
            description="""
                Box and Whisker plot
            """,
        )
        parser.add_argument(
            "-y",
            "--yearly",
            action="store_true",
            default=False,
            dest="year",
            help="Flag to show yearly bw plot",
        )
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            qa_view.display_bw(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                yearly=ns_parser.year,
            )

    @log_start_end(log=logger)
    def call_decompose(self, other_args: List[str]):
        """Process decompose command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="decompose",
            description="""
                Decompose time series as:
                - Additive Time Series = Level + CyclicTrend + Residual + Seasonality
                - Multiplicative Time Series = Level * CyclicTrend * Residual * Seasonality
            """,
        )
        parser.add_argument(
            "-m",
            "--multiplicative",
            action="store_true",
            default=False,
            dest="multiplicative",
            help="decompose using multiplicative model instead of additive",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_seasonal(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                multiplicative=ns_parser.multiplicative,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_cusum(self, other_args: List[str]):
        """Process cusum command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="cusum",
            description="""
                Cumulative sum algorithm (CUSUM) to detect abrupt changes in data
            """,
        )
        parser.add_argument(
            "-t",
            "--threshold",
            dest="threshold",
            type=float,
            default=(
                max(self.data[self.target].values) - min(self.data[self.target].values)
            )
            / 40,
            help="threshold",
        )
        parser.add_argument(
            "-d",
            "--drift",
            dest="drift",
            type=float,
            default=(
                max(self.data[self.target].values) - min(self.data[self.target].values)
            )
            / 80,
            help="drift",
        )
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            qa_view.display_cusum(
                data=self.data,
                target=self.target,
                threshold=ns_parser.threshold,
                drift=ns_parser.drift,
            )

    @log_start_end(log=logger)
    def call_acf(self, other_args: List[str]):
        """Process acf command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="acf",
            description="""
                Auto-Correlation and Partial Auto-Correlation Functions for diff and diff diff crypto data
            """,
        )
        parser.add_argument(
            "-l",
            "--lags",
            dest="lags",
            type=check_positive,
            default=15,
            help="maximum lags to display in plots",
        )
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            if self.target != "Close":
                console.print(
                    "Target not Close.  For best results, use `pick Close` first."
                )

            qa_view.display_acf(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                lags=ns_parser.lags,
            )

    @log_start_end(log=logger)
    def call_rolling(self, other_args: List[str]):
        """Process rolling command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="rolling",
            description="""
                Rolling mean and std deviation
            """,
        )
        parser.add_argument(
            "-w",
            "--window",
            action="store",
            dest="n_window",
            type=check_positive,
            default=14,
            help="Window length",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            rolling_view.display_mean_std(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                window=ns_parser.n_window,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_spread(self, other_args: List[str]):
        """Process spread command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="spread",
            description="""Shows rolling spread measurement
            """,
        )
        parser.add_argument(
            "-w",
            "--window",
            action="store",
            dest="n_window",
            type=check_positive,
            default=14,
            help="Window length",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            rolling_view.display_spread(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                window=ns_parser.n_window,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_quantile(self, other_args: List[str]):
        """Process quantile command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="quantile",
            description="""
                The quantiles are values which divide the distribution such that
                there is a given proportion of observations below the quantile.
                For example, the median is a quantile. The median is the central
                value of the distribution, such that half the points are less than
                or equal to it and half are greater than or equal to it.

                By default, q is set at 0.5, which effectively is median. Change q to
                get the desired quantile (0<q<1).
            """,
        )
        parser.add_argument(
            "-w",
            "--window",
            action="store",
            dest="n_window",
            type=check_positive,
            default=14,
            help="window length",
        )
        parser.add_argument(
            "-q",
            "--quantile",
            action="store",
            dest="f_quantile",
            type=check_proportion_range,
            default=0.5,
            help="quantile",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            rolling_view.display_quantile(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                window=ns_parser.n_window,
                quantile=ns_parser.f_quantile,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_skew(self, other_args: List[str]):
        """Process skew command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="skew",
            description="""
                Skewness is a measure of asymmetry or distortion of symmetric
                distribution. It measures the deviation of the given distribution
                of a random variable from a symmetric distribution, such as normal
                distribution. A normal distribution is without any skewness, as it is
                symmetrical on both sides. Hence, a curve is regarded as skewed if
                it is shifted towards the right or the left.
            """,
        )
        parser.add_argument(
            "-w",
            "--window",
            action="store",
            dest="n_window",
            type=check_positive,
            default=14,
            help="window length",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            rolling_view.display_skew(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                window=ns_parser.n_window,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_kurtosis(self, other_args: List[str]):
        """Process kurtosis command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="kurtosis",
            description="""
                Kurtosis is a measure of the "tailedness" of the probability distribution
                of a real-valued random variable. Like skewness, kurtosis describes the shape
                of a probability distribution and there are different ways of quantifying it
                for a theoretical distribution and corresponding ways of estimating it from
                a sample from a population. Different measures of kurtosis may have different
                interpretations.
            """,
        )
        parser.add_argument(
            "-w",
            "--window",
            action="store",
            dest="n_window",
            type=check_positive,
            default=14,
            help="window length",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            rolling_view.display_kurtosis(
                symbol=self.symbol,
                data=self.data,
                target=self.target,
                window=ns_parser.n_window,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_normality(self, other_args: List[str]):
        """Process normality command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="normality",
            description="""
                Normality tests
            """,
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_normality(
                data=self.data, target=self.target, export=ns_parser.export
            )

    @log_start_end(log=logger)
    def call_qqplot(self, other_args: List[str]):
        """Process qqplot command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="qqplot",
            description="""
                Display QQ plot vs normal quantiles
            """,
        )
        ns_parser = self.parse_known_args_and_warn(parser, other_args)
        if ns_parser:
            qa_view.display_qqplot(
                symbol=self.symbol, data=self.data, target=self.target
            )

    @log_start_end(log=logger)
    def call_unitroot(self, other_args: List[str]):
        """Process unitroot command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="unitroot",
            description="""
                Unit root test / stationarity (ADF, KPSS)
            """,
        )
        parser.add_argument(
            "-r",
            "--fuller_reg",
            help="Type of regression.  Can be ‘c’,’ct’,’ctt’,’nc’ 'c' - Constant and t - trend order",
            choices=self.FULLER_REG,
            default="c",
            type=str,
            dest="fuller_reg",
        )
        parser.add_argument(
            "-k",
            "--kps_reg",
            help="Type of regression.  Can be ‘c’,’ct'",
            choices=self.KPS_REG,
            type=str,
            dest="kpss_reg",
            default="c",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, export_allowed=EXPORT_ONLY_RAW_DATA_ALLOWED
        )
        if ns_parser:
            qa_view.display_unitroot(
                data=self.data,
                target=self.target,
                fuller_reg=ns_parser.fuller_reg,
                kpss_reg=ns_parser.kpss_reg,
                export=ns_parser.export,
            )
