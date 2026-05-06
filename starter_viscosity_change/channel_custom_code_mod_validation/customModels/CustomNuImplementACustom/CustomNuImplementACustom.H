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

Class
    Foam::laminarModels::generalisedNewtonianViscosityModels::
    CustomNuImplementACustom

Description
    Custom modified power-law generalised Newtonian viscosity model.

    Kinematic viscosity [m^2/s]:

        nu = nu_inf + k * max(gammaDot, gammaDotmin)^(n - 1)

    where gammaDot is the local strain rate (sqrt(2)*|symm(grad(U))|),
    nu_inf is the infinite-shear-rate viscosity, k is the consistency
    index (units m^2/s, i.e. the prefactor when gammaDot is non-dimensionalised
    by 1 s^-1), gammaDotmin is the minimum strain-rate clamp (1/s), and n
    is the power-law index (dimensionless).

    Example specification in constant/momentumTransport:
    \verbatim
        laminar
        {
            model            generalised Newtonian;
            viscosityModel   CustomNuImplementACustom;

            CustomNuImplementACustomCoeffs
            {
                nu_inf       1e-6;
                k            1e-3;
                gammaDotmin  1e-4;
                n            0.8;
            }
        }
    \endverbatim

SourceFiles
    CustomNuImplementACustom.C

\*---------------------------------------------------------------------------*/

#ifndef CustomNuImplementACustom_H
#define CustomNuImplementACustom_H

#include "strainRateViscosityModel.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace laminarModels
{
namespace generalisedNewtonianViscosityModels
{

/*---------------------------------------------------------------------------*\
              Class CustomNuImplementACustom Declaration
\*---------------------------------------------------------------------------*/

class CustomNuImplementACustom
:
    public strainRateViscosityModel
{
    // Private Data

        //- Infinite-shear-rate kinematic viscosity [m^2/s]
        dimensionedScalar nuInf_;

        //- Consistency index [m^2/s] (prefactor; gammaDot non-dimensionalised
        //  by 1 s^-1 inside nu())
        dimensionedScalar k_;

        //- Minimum strain-rate clamp [1/s]
        dimensionedScalar gammaDotMin_;

        //- Power-law index [-]
        dimensionedScalar n_;


public:

    //- Runtime type information
    TypeName("CustomNuImplementACustom");


    // Constructors

        //- Construct from components
        CustomNuImplementACustom
        (
            const dictionary& viscosityProperties,
            const Foam::viscosity& viscosity,
            const volVectorField& U
        );


    //- Destructor
    virtual ~CustomNuImplementACustom()
    {}


    // Member Functions

        //- Read transportProperties dictionary
        virtual bool read(const dictionary& viscosityProperties);

        //- Return the laminar viscosity
        virtual tmp<volScalarField> nu
        (
            const volScalarField& nu0,
            const volScalarField& strainRate
        ) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace generalisedNewtonianViscosityModels
} // End namespace laminarModels
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
