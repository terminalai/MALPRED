CATEGORICAL_FEATURES = [
    'ProductName', 'Platform', 'Processor', 'OsVer', 'OsPlatformSubRelease', 'SkuEdition', 'SmartScreen',
    'Census_MDC2FormFactor', 'Census_DeviceFamily', 'Census_PrimaryDiskTypeName', 'Census_ChassisTypeName',
    'Census_PowerPlatformRoleName', 'Census_OSVersion', 'Census_OSArchitecture', 'Census_OSBranch', 'Census_OSEdition',
    'Census_OSSkuName', 'Census_OSInstallTypeName', 'Census_OSWUAutoUpdateOptionsName', 'Census_GenuineStateName',
    'Census_ActivationChannel', 'Census_FlightRing', 'OsBuildLab1', 'OsBuildLab2', 'OsBuildLab3', 'OsBuildLab4']

NUMERIC_FEATURES = [
    'IsBeta', 'RtpStateBitfield', 'IsSxsPassiveMode', 'AVProductStatesIdentifier', 'AVProductsInstalled',
    'AVProductsEnabled', 'HasTpm', 'CountryIdentifier', 'CityIdentifier', 'OrganizationIdentifier', 'GeoNameIdentifier',
    'LocaleEnglishNameIdentifier', 'OsBuild', 'OsSuite', 'IsProtected', 'AutoSampleOptIn', 'SMode', 'IeVerIdentifier',
    'Firewall', 'UacLuaenable', 'Census_OEMNameIdentifier', 'Census_OEMModelIdentifier', 'Census_ProcessorCoreCount',
    'Census_ProcessorManufacturerIdentifier', 'Census_ProcessorModelIdentifier', 'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity', 'Census_HasOpticalDiskDrive', 'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches', 'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical', 'Census_InternalBatteryNumberOfCharges', 'Census_OSBuildNumber',
    'Census_OSBuildRevision', 'Census_OSInstallLanguageIdentifier', 'Census_OSUILocaleIdentifier',
    'Census_IsPortableOperatingSystem', 'Census_IsFlightsDisabled', 'Census_ThresholdOptIn',
    'Census_FirmwareManufacturerIdentifier', 'Census_FirmwareVersionIdentifier', 'Census_IsSecureBootEnabled',
    'Census_IsWIMBootEnabled', 'Census_IsVirtualDevice', 'Census_IsTouchEnabled', 'Census_IsPenCapable',
    'Census_IsAlwaysOnAlwaysConnectedCapable', 'Wdft_IsGamer', 'Wdft_RegionIdentifier', 'OsBuildLab0',
    'Census_OSVersion1', 'Census_OSVersion2', 'Census_OSVersion3', 'AvSigVersion0', 'AvSigVersion1', 'AvSigVersion2',
    'AvSigVersion3', 'AppVersion0', 'AppVersion1', 'AppVersion2', 'AppVersion3', 'EngineVersion0', 'EngineVersion1',
    'EngineVersion2', 'EngineVersion3'
]

NUM_CATEGORIES = {
    'ProductName': 6, 'Platform': 4, 'Processor': 3, 'OsVer': 58, 'OsPlatformSubRelease': 9, 'SkuEdition': 8,
    'SmartScreen': 21, 'Census_MDC2FormFactor': 13, 'Census_DeviceFamily': 3, 'Census_PrimaryDiskTypeName': 4,
    'Census_ChassisTypeName': 52, 'Census_PowerPlatformRoleName': 10, 'Census_OSVersion': 469,
    'Census_OSArchitecture': 3, 'Census_OSBranch': 32, 'Census_OSEdition': 33, 'Census_OSSkuName': 30,
    'Census_OSInstallTypeName': 9, 'Census_OSWUAutoUpdateOptionsName': 6, 'Census_GenuineStateName': 5,
    'Census_ActivationChannel': 6, 'Census_FlightRing': 10, 'OsBuildLab1': 284, 'OsBuildLab2': 3, 'OsBuildLab3': 51,
    'OsBuildLab4': 367
}

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
