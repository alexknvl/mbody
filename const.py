# Mathematical constants.
kPi = 3.141592653589793238462643383279502884

# Fundamental constants.
kLightSpeed = 299792458
kReducedPlanckConstant = 1.5457172647e-34
kElementaryCharge = 1.60217656535e-19

# SI electromagnetism constants.
kMagneticConstant = 4 * kPi * 1e-7
kCoulumbConstant = kLightSpeed * kLightSpeed * kMagneticConstant / (4 * kPi)
kElectricConstant = 1 / (kMagneticConstant * kLightSpeed * kLightSpeed)

# Atomic masses.
kAtomicMassUnit = 1.66053892173e-27
kElectronMass = (1 / 1822.8884845) * kAtomicMassUnit
kProtonMass = 1.007276466812 * kAtomicMassUnit
kNeutronMass = 1.00866491600 * kAtomicMassUnit
kHydrogenMass = 1.0081 * kAtomicMassUnit
kLithiumMass = 6.9412 * kAtomicMassUnit

# Other constants.
kBohrMagneton = kElementaryCharge * kReducedPlanckConstant / (2 * kElectronMass)
kNuclearMagneton = kElementaryCharge * kReducedPlanckConstant / (2 * kProtonMass)
kBoltzmannConstant = 1.3806488e-23
