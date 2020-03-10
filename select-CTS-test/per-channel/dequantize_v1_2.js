// Generated file (from: dequantize_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Dequantize v1_2 example-1', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_first_dim
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input04_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output04_expect = [-256, -254, -252, -250, -248, -246, -244, -242, -240, -238, -236, -234, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5];

    let type8 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type9_length = product(type9.dimensions);

    let input04 = operandIndex++;
    model.addOperand(type8);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
    let output04 = operandIndex++;
    model.addOperand(type9);

    model.addOperation(nn.DEQUANTIZE, [input04], [output04]);

    model.identifyInputsAndOutputs([input04], [output04]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input04_input = new Int8Array(input04_value);
    execution.setInput(0, input04_input);
    let output04_output = new Float32Array(type9_length);
    execution.setOutput(0, output04_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(output04_output[i], output04_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-2', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_first_dim_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input04_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output04_expect = [-256, -254, -252, -250, -248, -246, -244, -242, -240, -238, -236, -234, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5];

    let type8 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type8_length = product(type8.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type9_length = product(type9.dimensions);

    let input04 = operandIndex++;
    model.addOperand(type8);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
    let output04 = operandIndex++;
    model.addOperand(type9);

    model.addOperation(nn.DEQUANTIZE, [input04], [output04]);

    model.identifyInputsAndOutputs([input04], [output04]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input04_input = new Int8Array(input04_value);
    execution.setInput(0, input04_input);
    let output04_output = new Float32Array(type9_length);
    execution.setOutput(0, output04_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(output04_output[i], output04_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-3', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_first_dim_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input04_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output04_expect = [-256.0, -254.0, -252.0, -250.0, -248.0, -246.0, -244.0, -242.0, -240.0, -238.0, -236.0, -234.0, 58.0, 58.5, 59.0, 59.5, 60.0, 60.5, 61.0, 61.5, 62.0, 62.5, 63.0, 63.5];

    let type29 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type29_length = product(type29.dimensions);
    let type8 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type8_length = product(type8.dimensions);

    let input04 = operandIndex++;
    model.addOperand(type8);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
    let output04 = operandIndex++;
    model.addOperand(type29);

    model.addOperation(nn.DEQUANTIZE, [input04], [output04]);

    model.identifyInputsAndOutputs([input04], [output04]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input04_input = new Int8Array(input04_value);
    execution.setInput(0, input04_input);
    let output04_output = new Float32Array(type29_length);
    execution.setOutput(0, output04_output);

    await execution.startCompute();

    for (let i = 0; i < type29_length; ++i) {
      assert.isTrue(almostEqualCTS(output04_output[i], output04_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-4', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_second_dim
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input05_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output05_expect = [-256.0, -254.0, -252.0, -250.0, -124.0, -123.0, -122.0, -121.0, -60.0, -59.5, -59.0, -58.5, 232.0, 234.0, 236.0, 238.0, 120.0, 121.0, 122.0, 123.0, 62.0, 62.5, 63.0, 63.5];

    let type10 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type10_length = product(type10.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type9_length = product(type9.dimensions);

    let input05 = operandIndex++;
    model.addOperand(type10);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
    let output05 = operandIndex++;
    model.addOperand(type9);

    model.addOperation(nn.DEQUANTIZE, [input05], [output05]);

    model.identifyInputsAndOutputs([input05], [output05]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input05_input = new Int8Array(input05_value);
    execution.setInput(0, input05_input);
    let output05_output = new Float32Array(type9_length);
    execution.setOutput(0, output05_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(output05_output[i], output05_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-5', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_second_dim_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input05_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output05_expect = [-256.0, -254.0, -252.0, -250.0, -124.0, -123.0, -122.0, -121.0, -60.0, -59.5, -59.0, -58.5, 232.0, 234.0, 236.0, 238.0, 120.0, 121.0, 122.0, 123.0, 62.0, 62.5, 63.0, 63.5];

    let type10 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type10_length = product(type10.dimensions);
    let type9 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type9_length = product(type9.dimensions);

    let input05 = operandIndex++;
    model.addOperand(type10);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
    let output05 = operandIndex++;
    model.addOperand(type9);

    model.addOperation(nn.DEQUANTIZE, [input05], [output05]);

    model.identifyInputsAndOutputs([input05], [output05]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input05_input = new Int8Array(input05_value);
    execution.setInput(0, input05_input);
    let output05_output = new Float32Array(type9_length);
    execution.setOutput(0, output05_output);

    await execution.startCompute();

    for (let i = 0; i < type9_length; ++i) {
      assert.isTrue(almostEqualCTS(output05_output[i], output05_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-6', async function() {
    // For 'Dequantize v1_2' example: examples_3d_per_channel_second_dim_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input05_value = [-128, -127, -126, -125, -124, -123, -122, -121, -120, -119, -118, -117, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127];
    let output05_expect = [-256.0, -254.0, -252.0, -250.0, -124.0, -123.0, -122.0, -121.0, -60.0, -59.5, -59.0, -58.5, 232.0, 234.0, 236.0, 238.0, 120.0, 121.0, 122.0, 123.0, 62.0, 62.5, 63.0, 63.5];

    let type10 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [2, 3, 4]};
    let type10_length = product(type10.dimensions);
    let type29 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 3, 4]};
    let type29_length = product(type29.dimensions);

    let input05 = operandIndex++;
    model.addOperand(type10);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
    let output05 = operandIndex++;
    model.addOperand(type29);

    model.addOperation(nn.DEQUANTIZE, [input05], [output05]);

    model.identifyInputsAndOutputs([input05], [output05]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input05_input = new Int8Array(input05_value);
    execution.setInput(0, input05_input);
    let output05_output = new Float32Array(type29_length);
    execution.setOutput(0, output05_output);

    await execution.startCompute();

    for (let i = 0; i < type29_length; ++i) {
      assert.isTrue(almostEqualCTS(output05_output[i], output05_expect[i]));
    }
  });
});
