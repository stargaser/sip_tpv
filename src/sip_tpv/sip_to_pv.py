# Licensed under a 3-clause BSD style license - see LICENSE.txt
"""
Convert SIP convention distortion keywords to the TPV convention.

This module includes the equations for converting from the SIP distortion representation
 to the PV or TPV representations, following the SPIE proceedings paper at
     http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1363103
     and http://web.ipac.caltech.edu/staff/shupe/reprints/SIP_to_PV_SPIE2012.pdf .
The work described in that paper is extended to 7th order.

Copyright (c) 2012-2017, California Institute of Technology

If you make use of this work, please cite:
"More flexibility in representing geometric distortion in astronomical images,"
  Shupe, David L.; Laher, Russ R.; Storrie-Lombardi, Lisa; Surace, Jason; Grillmair, Carl;
  Levitan, David; Sesar, Branimir, 2012, in Software and Cyberinfrastructure for
  Astronomy II. Proceedings of the SPIE, Volume 8451, article id. 84511M.

Thanks to Octavi Fors for contributing code modifications for better modularization,
   and for extensive testing.

Funding is acknowledged from NASA to the NASA Herschel Science Center and the
   Spitzer Science Center.

Contact: David Shupe, Caltech/IPAC.

"""

from copy import copy
from pathlib import Path

import click
import astropy.io.fits as fits

from ._cli import EXTENSION, get_hdu
from .pvsiputils import (
    sym_tpvexprs,
    get_sip_keywords,
    real_sipexprs,
    add_pv_keywords,
    remove_sip_keywords,
)


def sip_to_pv(header, tpv_format=True, preserve=False):
    """ Function which wraps the sip_to_pv conversion

    Parameters:
    -----------
    header (fits.Header) : header of file with SIP convention keywords
    tpv_format (boolean) : modify CTYPE1 and CTYPE2 to TPV convention RA---TPV, DEC--TPV
    preserve (boolean) : preserve the SIP keywords in the header (default is to delete)

    Returns:
    --------
    None (header is modified in-place)
    """
    pvrange, tpvx, tpvy = sym_tpvexprs()
    cd, ac, bc = get_sip_keywords(header)
    sipx, sipy = real_sipexprs(cd, ac, bc)
    add_pv_keywords(header, sipx, sipy, pvrange, tpvx, tpvy,
                    int(header['B_ORDER']))
    if not preserve:
        remove_sip_keywords(header)


@click.command()
@click.argument("infile", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.argument("outfile", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--extension", default=0, type=EXTENSION,
              help="Index or EXTNAME of the HDU with the SIP header (default 0).")
@click.option("--overwrite", is_flag=True,
              help="Overwrite output file if it exists.")
@click.option("--write-tan", "--write_tan", is_flag=True,
              help="Write -TAN in CTYPE keywords, not -TPV.")
@click.option("--write-tan-sip", "--write_tan_sip", is_flag=True,
              help="Write -TAN-SIP in CTYPE keywords, not -TPV.")
@click.option("--preserve-sip", "--preserve_sip", is_flag=True,
              help="Retain SIP keywords in header.")
def cli(infile, outfile, extension, overwrite, write_tan, write_tan_sip, preserve_sip):
    """Convert FITS files with SIP distortion to TPV representation."""
    if write_tan and write_tan_sip:
        raise click.UsageError("--write-tan and --write-tan-sip are mutually exclusive.")
    with fits.open(infile) as hdu:
        header = copy(get_hdu(hdu, extension).header)
        sip_to_pv(header, tpv_format=not write_tan, preserve=preserve_sip)
        if write_tan_sip:
            header['CTYPE1'] = 'RA---TAN-SIP'
            header['CTYPE2'] = 'DEC--TAN-SIP'
        hdu[extension].header = header
        hdu.writeto(outfile, overwrite=overwrite)


if __name__ == "__main__":
    cli()
