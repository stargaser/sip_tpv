from copy import copy
from pathlib import Path

import click
import astropy.io.fits as fits

from .pvsiputils import (
    get_pv_keywords,
    sym_sipexprs,
    real_tpvexprs,
    add_sip_keywords,
    remove_pv_keywords,
)

# Licensed under a 3-clause BSD style license - see LICENSE.txt
"""
Convert TPV convention distortion keywords to the SIP convention.

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

version = "2.0"


def pv_to_sip(header, preserve=False, add_reverse=True,
              aporder=None, bporder=None):
    """Function which wraps the PV to SIP conversion."""
    cd, pv1, pv2 = get_pv_keywords(header)
    sipu, sipv = sym_sipexprs()
    tpvu, tpvv = real_tpvexprs(cd, pv1, pv2)
    add_sip_keywords(header, tpvu, tpvv, sipu, sipv, add_reverse,
                     aporder, bporder)
    if not preserve:
        remove_pv_keywords(header)


@click.command()
@click.argument("infile", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.argument("outfile", type=click.Path(path_type=Path, dir_okay=False))
@click.option("--extension", default=0, type=int,
              help="Extension of file with PV header (default 0).")
@click.option("--overwrite", is_flag=True,
              help="Overwrite output file if it exists.")
@click.option("--preserve-tpv", is_flag=True,
              help="Retain PV keywords in header.")
@click.option("--add-reverse", is_flag=True,
              help="Write the reverse SIP coefficients.")
@click.option("--aporder", default=None, type=int,
              help="Order for reverse polynomial, axis 1 (defaults to A_ORDER).")
@click.option("--bporder", default=None, type=int,
              help="Order for reverse polynomial, axis 2 (defaults to B_ORDER).")
def cli(infile, outfile, extension, overwrite, preserve_tpv, add_reverse,
        aporder, bporder):
    """Convert FITS files with TPV distortion to SIP representation."""
    with fits.open(infile) as hdu:
        header = copy(hdu[extension].header)
        pv_to_sip(header, preserve=preserve_tpv, add_reverse=add_reverse,
                  aporder=aporder, bporder=bporder)
        hdu[extension].header = header
        hdu.writeto(outfile, overwrite=overwrite)


if __name__ == "__main__":
    cli()
