import argparse
import logging
from typing import List

# flake8: noqa

from openbb_terminal.custom_prompt_toolkit import NestedCompleter

from openbb_terminal import feature_flags as obbff
from openbb_terminal.cryptocurrency.nft import (
    nftpricefloor_model,
    nftpricefloor_view,
    opensea_view,
)
from openbb_terminal.decorators import log_start_end
from openbb_terminal.helper_funcs import (
    EXPORT_ONLY_RAW_DATA_ALLOWED,
)
from openbb_terminal.menu import session
from openbb_terminal.parent_classes import BaseController
from openbb_terminal.rich_config import console, MenuText

logger = logging.getLogger(__name__)


class NFTController(BaseController):
    """NFT Controller class"""

    CHOICES_COMMANDS = [
        "stats",
        "collections",
        "fp",
    ]
    PATH = "/crypto/nft/"

    def __init__(self, queue: List[str] = None):
        """Constructor"""
        super().__init__(queue)

        nft_price_floor_collections = nftpricefloor_model.get_collection_slugs()

        if session and obbff.USE_PROMPT_TOOLKIT:
            choices: dict = {c: {} for c in self.controller_choices}

            choices["support"] = self.SUPPORT_CHOICES
            choices["about"] = self.ABOUT_CHOICES
            choices["fp"] = {c: {} for c in nft_price_floor_collections}
            choices["fp"]["--raw"] = {}
            choices["fp"]["--limit"] = {str(c): {} for c in range(1, 100)}
            choices["fp"]["-l"] = "--limit"
            choices["stats"] = {
                "--slug": None,
                "-s": None,
            }
            choices["collections"] = {
                "--fp": {},
                "--sales": {},
                "--limit": {str(c): {} for c in range(1, 50)},
                "-l": "--limit",
            }

            self.completer = NestedCompleter.from_nested_dict(choices)

    def print_help(self):
        """Print help"""
        mt = MenuText("crypto/nft/", 70)
        mt.add_cmd("stats")
        mt.add_cmd("fp")
        mt.add_cmd("collections")
        console.print(text=mt.menu_text, menu="Cryptocurrency - Non Fungible Token")

    @log_start_end(log=logger)
    def call_fp(self, other_args: List[str]):
        """Process fp command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="fp",
            description="""
                Display floor price of a certain NFT collection.
                [Source: https://nftpricefloor.com/]
            """,
        )

        parser.add_argument(
            "-s",
            "--slug",
            type=str,
            help="NFT floor price collection slug (e.g., bored-ape-yacht-club)",
            dest="slug",
            required="-h" not in other_args,
        )
        if other_args and not other_args[0][0] == "-":
            other_args.insert(0, "--slug")

        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, EXPORT_ONLY_RAW_DATA_ALLOWED, raw=True, limit=10
        )

        if ns_parser:
            nftpricefloor_view.display_floor_price(
                slug=ns_parser.slug,
                export=ns_parser.export,
                raw=ns_parser.raw,
                limit=ns_parser.limit,
            )

    @log_start_end(log=logger)
    def call_stats(self, other_args: List[str]):
        """Process stats command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="info",
            description="""
                Display stats about an opensea nft collection. e.g. alien-frens
                [Source: https://nftpricefloor.com/]
            """,
        )

        parser.add_argument(
            "-s",
            "--slug",
            type=str,
            help="Opensea collection slug (e.g., mutant-ape-yacht-club)",
            dest="slug",
            required="-h" not in other_args,
        )
        if other_args and not other_args[0][0] == "-":
            other_args.insert(0, "--slug")

        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, EXPORT_ONLY_RAW_DATA_ALLOWED
        )

        if ns_parser:
            opensea_view.display_collection_stats(
                slug=ns_parser.slug,
                export=ns_parser.export,
            )

    @log_start_end(log=logger)
    def call_collections(self, other_args: List[str]):
        """Process collections command"""
        parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="collections",
            description="NFT Collections [Source: https://nftpricefloor.com/]",
        )
        parser.add_argument(
            "--fp",
            dest="fp",
            action="store_true",
            default=False,
            help="Flag to display floor price over time for top collections",
        )
        parser.add_argument(
            "--sales",
            dest="sales",
            action="store_true",
            default=False,
            help="Flag to display sales over time for top collections",
        )
        ns_parser = self.parse_known_args_and_warn(
            parser, other_args, EXPORT_ONLY_RAW_DATA_ALLOWED, limit=5
        )
        if ns_parser:
            nftpricefloor_view.display_collections(
                show_sales=ns_parser.sales,
                show_fp=ns_parser.fp,
                num=ns_parser.limit,
                export=ns_parser.export,
            )
