"""Cboe options view"""
__docformat__ = "numpy"

import logging
import os
import re
from bisect import bisect_left
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from openpyxl import Workbook
from scipy.stats import binom

import openbb_terminal.config_plot as cfp
from openbb_terminal.config_plot import PLOT_DPI
from openbb_terminal.config_terminal import theme
from openbb_terminal.core.config.paths import MISCELLANEOUS_DIRECTORY
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    excel_columns,
    export_data,
    get_rf,
    is_valid_axes_count,
    plot_autoscale,
    print_rich_table,
)
from openbb_terminal.rich_config import console
from openbb_terminal.stocks.options import op_helpers, cboe_model
from openbb_terminal.stocks.options.cboe_model import (
    get_expirations,
    get_option_chain,
)

# pylint: disable=C0302


logger = logging.getLogger(__name__)


def header_fmt(header: str) -> str:
    """
    Formats strings to appear as titles

    Parameters
    ----------
    header: str
        The string to be formatted

    Returns
    ----------
    new_header: str
        The clean string to use as a header
    """

    words = re.findall("[A-Z][^A-Z]*", header)
    if words == []:
        words = [header]
    new_header = " ".join(words)
    new_header = new_header.replace("_", " ")
    return new_header.title()

log_start_end(log=logger)
def display_chains(
    symbol: str,
    expiry: str,
    min_sp: float = -1,
    max_sp: float = -1,
    calls_only: bool = False,
    puts_only: bool = False,
    export: str = "",
):
    """Display option chains for given ticker and expiration

    Parameters
    ----------
    symbol: str
        Stock ticker symbol
    expiry: str
        Expiration for option chain
    min_sp: float
        Min strike
    max_sp: float
        Max strike
    calls_only: bool
        Flag to get calls only
    puts_only: bool
        Flag to get puts only
    export: str
        Format to export data

    """
    # Logic for converting calls/puts into "get calls/puts"
    call_bool = True
    put_bool = True
    if calls_only:
        call_bool = True
        put_bool = False
    if puts_only:
        call_bool = False
        put_bool = True

    option_chains = cboe_model.get_option_chain(
        symbol=symbol,
        expiry=expiry,
        calls=call_bool,
        puts=put_bool,
        min_sp=min_sp,
        max_sp=max_sp,
    ).fillna("-")
    if option_chains.empty:
        console.print("[red]Option chains not found.[/red]")
        return

    # There are 3 possibilities.  Calls only, puts only or both.
    # If calls only or puts only, we are actually set
    # because the columns are nicely named
    if calls_only or puts_only:
        title = "Call " if calls_only else "Put "
        print_rich_table(
            option_chains,
            title=title + "Option Chain (15 min delayed) (Greeks calculated by Cboe)",
            floatfmt=[
                ".2f",
                ".2f",
                ".2f",
                ".2f",
                ".0f",
                ".0f",
                ".3f",
                ".3f",
                ".3f",
                ".3f",
            ],
            headers=[
                "Strike",
                "Last Price",
                "Bid",
                "Ask",
                "Volume",
                "Open Interest",
                "IV",
                "Delta",
                "Gamma",
                "Theta",
            ],
        )

    # Put the columns into the order for showing them
    if calls_only and puts_only:
        option_chains = option_chains[
            [
                "impliedVolatility_call",
                "Theta_call",
                "Gamma_call",
                "Delta_call",
                "volume_call",
                "openInterest_call",
                "bid_call",
                "ask_call",
                "lastPrice_call",
                "strike",
                "lastPrice_put",
                "ask_put",
                "bid_put",
                "openInterest_put",
                "volume_put",
                "Delta_put",
                "Gamma_put",
                "Theta_put",
                "impliedVolatility_put",
            ]
        ]

        # In order to add color to call/put, the numbers will have to be strings.
        # So floatfmt will not work in print_rich_table, so lets format them now.

        float_fmt = [
            ".3f",
            ".3f",
            ".3f",
            ".3f",
            ".0f",
            ".0f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".2f",
            ".0f",
            ".0f",
            ".3f",
            ".3f",
            ".3f",
            ".3f",
        ]
        # pylint: disable=W0640

        for idx, fmt in enumerate(float_fmt):
            option_chains.iloc[:, idx] = option_chains.iloc[:, idx].apply(
                lambda x: str("{:" + fmt + "}").format(float(x)) if x != "-" else x
            )
        # pylint: enable=W0640

    # Make anything _call green and anything _put red
    for col in option_chains.columns:
        if col.endswith("_call"):
            option_chains[col] = option_chains[col].apply(
                lambda x: f"[green]{x}[/green]"
            )
        if col.endswith("_put"):
            option_chains[col] = option_chains[col].apply(lambda x: f"[red]{x}[/red]")

    print_rich_table(
        option_chains,
        title=f"Yahoo Option Chain (15 min delayed) for {expiry} (Greeks calculated by Cboe)",
        headers=[header_fmt(x) for x in option_chains.columns],
    )
    export_data(
        export,
        os.path.dirname(os.path.abspath(__file__)),
        "chains_yf",
        option_chains,
    )
