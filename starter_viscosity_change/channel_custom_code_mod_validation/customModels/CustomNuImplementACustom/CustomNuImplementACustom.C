/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2018-2021 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "CustomNuImplementACustom.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarModels
{
namespace generalisedNewtonianViscosityModels
{
    defineTypeNameAndDebug(CustomNuImplementACustom, 0);

    addToRunTimeSelectionTable
    (
        generalisedNewtonianViscosityModel,
        CustomNuImplementACustom,
        dictionary
    );
}
}
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::laminarModels::generalisedNewtonianViscosityModels::
CustomNuImplementACustom::CustomNuImplementACustom
(
    const dictionary& viscosityProperties,
    const Foam::viscosity& viscosity,
    const volVectorField& U
)
:
    strainRateViscosityModel(viscosityProperties, viscosity, U),
    nuInf_("nu_inf", dimViscosity, 0.0),
    k_("k", dimViscosity, 0.0),
    gammaDotMin_("gammaDotmin", dimless/dimTime, small),
    n_("n", dimless, 1.0)
{
    read(viscosityProperties);
    correct();
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

bool
Foam::laminarModels::generalisedNewtonianViscosityModels::
CustomNuImplementACustom::read
(
    const dictionary& viscosityProperties
)
{
    strainRateViscosityModel::read(viscosityProperties);

    const dictionary& coeffs =
        viscosityProperties.optionalSubDict(typeName + "Coeffs");

    nuInf_.read(coeffs);
    k_.read(coeffs);
    gammaDotMin_.read(coeffs);
    n_.read(coeffs);

    return true;
}


Foam::tmp<Foam::volScalarField>
Foam::laminarModels::generalisedNewtonianViscosityModels::
CustomNuImplementACustom::nu
(
    const volScalarField& nu0,
    const volScalarField& strainRate
) const
{
    // Non-dimensionalise the strain rate by 1 s^-1 so that pow() receives
    // a dimensionless argument, matching the pattern used in powerLaw.C.
    // gammaDotMin_ has units 1/s; multiplying by dimTime (1 s) makes it
    // dimensionless for the max() comparison.
    const dimensionedScalar tRef(dimTime, 1.0);

    const volScalarField gammaDotDimless
    (
        max
        (
            tRef * strainRate,
            tRef * gammaDotMin_
        )
    );

    return
        nuInf_
      + k_ * pow(gammaDotDimless, n_.value() - scalar(1));
}


// ************************************************************************* //
