"""Helpers shared by the pv_to_sip and sip_to_pv command-line tools."""

import click


class ExtensionType(click.ParamType):
    """An HDU given either as an integer index or as an EXTNAME."""

    name = "INDEX|NAME"

    def convert(self, value, param, ctx):
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except ValueError:
            return value


EXTENSION = ExtensionType()


def get_hdu(hdulist, extension):
    """Return hdulist[extension], or exit with a usage error if absent."""
    try:
        return hdulist[extension]
    except (IndexError, KeyError):
        raise click.BadParameter(
            f"no extension {extension!r} in input file "
            f"({len(hdulist)} HDUs)", param_hint="'--extension'")
