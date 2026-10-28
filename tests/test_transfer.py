import pandas as pd
from iaa import transfer


def test_harmonise_keeps_ids_and_maps_age_bands():
    df = pd.DataFrame({"row_id": ["M001"], "driver_age": [24], "cover_type": [" Comprehensive "]})
    h = transfer.harmonise(df, "motor")
    assert h.row_id.iloc[0] == "M001"                 # IDs must never be altered (bug caught 28 Oct)
    assert h.driver_age_band.iloc[0] == "17-24"
    assert h.cover_type.iloc[0] == "comprehensive"
    assert transfer.harmonise(pd.DataFrame({"driver_age": [25, 64, 65]}), "motor").driver_age_band.tolist() == ["25-34", "50-64", "65+"]
