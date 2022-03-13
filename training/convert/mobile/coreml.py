import coremltools as ct


def convert2coreml(model, save_path, scale, bias):
    # inference is supported only on MacOS
    # cpnvertation is not supported on Windows, use WSL
    size = model.inputs[0].shape[1:3]
    inputs = ct.ImageType(
        name="input_1", shape=(ct.RangeDim(), *size, 3), scale=scale, bias=bias
    )
    coreml_model = ct.convert(model, inputs=[inputs])
    spec = coreml_model.get_spec()
    ct.utils.save_spec(spec, save_path)
