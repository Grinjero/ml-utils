

def asfreq_0_holes(group, time_column, freq, columns_to_0):
    asf = group.set_index(time_column, drop=True).sort_index().asfreq(freq)
    if columns_to_0 is not None:
        asf.loc[:, columns_to_0] = asf.loc[:, columns_to_0].fillna(0)

    columns_others = list(set(asf.columns) - set(columns_to_0))
    asf.loc[:, columns_others] = asf.loc[:, columns_others].ffill()
    return asf.reset_index(drop=False)


def resample_0_holes(group, time_column, freq, columns_to_0):
    columns_others = set(group.columns) - set(columns_to_0)
    columns_others = columns_others - set([time_column])
    columns_others = list(columns_others)
    asf = group.set_index(time_column, drop=True).sort_index().resample(freq)
    agg_rules = {}
    if columns_to_0 is not None:
        for col_to_0 in columns_to_0:
            agg_rules[col_to_0] = "sum"
        # asf.loc[:, columns_to_0] = asf.loc[:, columns_to_0].fillna(0)
    for col_other in columns_others:
        agg_rules[col_other] = "first"

    asf = asf.agg(agg_rules)
    asf.loc[:, columns_others] = asf.loc[:, columns_others].ffill()
    return asf.reset_index(drop=False)
