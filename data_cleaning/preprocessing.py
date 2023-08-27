import pandas as pd


def cull_columns(data: pd.DataFrame):
    columns_to_be_removed = []
    percent = (data.isnull().sum() / data.shape[0]) * 100

    for col in data.columns:
        if percent.loc[col] >= 70:
            columns_to_be_removed.append(col)

    data.drop(columns=columns_to_be_removed, inplace=True)


def clean_df(train: pd.DataFrame) -> pd.DataFrame:
    train[
        ['OsBuildLab0', 'OsBuildLab1', 'OsBuildLab2', 'OsBuildLab3', 'OsBuildLab4']
    ] = train.loc[:, 'OsBuildLab'].str.split('.', expand=True)

    train.loc[:, 'OsBuildLab0'] = train['OsBuildLab0'].astype(str)

    train.loc[train['OsBuildLab0'] == 'nan', 'OsBuildLab0'] = 0

    train['OsBuildLab0'] = train['OsBuildLab0'].astype(int)

    train.loc[:,
    ['OsBuildLab1', 'OsBuildLab2', 'OsBuildLab3', 'OsBuildLab4']
    ] = train.loc[:, ['OsBuildLab1', 'OsBuildLab2', 'OsBuildLab3', 'OsBuildLab4']].astype('category')

    train['AvSigVersion'] = train['AvSigVersion'].astype(str)

    train.loc[train.AvSigVersion.str.contains('2&#x17;3'), 'AvSigVersion'] = '1.2173.1144.0'

    train[
        ['Census_OSVersion0', 'Census_OSVersion1', 'Census_OSVersion2', 'Census_OSVersion3']
    ] = train.loc[:, 'Census_OSVersion'].str.split('.', expand=True)

    train[
        ['AvSigVersion0', 'AvSigVersion1', 'AvSigVersion2', 'AvSigVersion3']
    ] = train.loc[:, 'AvSigVersion'].str.split('.', expand=True)

    train[
        ['AppVersion0', 'AppVersion1', 'AppVersion2', 'AppVersion3']
    ] = train.loc[:, 'AppVersion'].str.split('.', expand=True)

    train[
        ['EngineVersion0', 'EngineVersion1', 'EngineVersion2', 'EngineVersion3']
    ] = train.loc[:, 'EngineVersion'].str.split('.', expand=True)

    train.loc[5244810, ['AvSigVersion0', 'AvSigVersion1', 'AvSigVersion2', 'AvSigVersion3']] = [0, 0, 0, 0]

    train.iloc[:, -16:] = train.iloc[:, -16:].astype(int)

    train.drop(
        ['MachineIdentifier', 'Census_OSVersion0', 'OsBuildLab', 'EngineVersion', 'AppVersion', 'AvSigVersion',
         'PuaMode', 'Census_ProcessorClass', 'DefaultBrowsersIdentifier', 'Census_IsFlightingInternal',
         'Census_InternalBatteryType'],
        inplace=True
    )

    cull_columns(train)

    return train



