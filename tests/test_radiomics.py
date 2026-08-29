from icvs_gbm_pfs.radiomics import _radiomic_feature_name


def test_shape_features_are_modality_independent() -> None:
    name = "original_shape_MeshVolume"
    assert _radiomic_feature_name("t1", name) == "shape__original_shape_MeshVolume"
    assert _radiomic_feature_name("ce_t1", name) == "shape__original_shape_MeshVolume"


def test_intensity_and_texture_features_retain_modality() -> None:
    assert (
        _radiomic_feature_name("flair", "wavelet-HLL_glcm_Contrast")
        == "flair__wavelet-HLL_glcm_Contrast"
    )
