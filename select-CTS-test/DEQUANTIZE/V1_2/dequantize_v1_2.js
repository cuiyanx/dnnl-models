// Generated file (from: dequantize_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Dequantize v1_2 example-1', async function() {
    // For 'Dequantize v1_2' example: examples_1d_quant8_asymm
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output0_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [10], scale: 0.5, zeroPoint: 127};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type1_length = product(type1.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type0);
    let output0 = operandIndex++;
    model.addOperand(type1);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Uint8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type1_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-2', async function() {
    // For 'Dequantize v1_2' example: examples_1d_quant8_asymm_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output0_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [10], scale: 0.5, zeroPoint: 127};
    let type0_length = product(type0.dimensions);
    let type1 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type1_length = product(type1.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type0);
    let output0 = operandIndex++;
    model.addOperand(type1);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Uint8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type1_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type1_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-3', async function() {
    // For 'Dequantize v1_2' example: examples_1d_quant8_asymm_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input0_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output0_expect = [-63.5, -63.0, -62.5, -62.0, -61.5, 62.0, 62.5, 63.0, 63.5, 64.0];

    let type0 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [10], scale: 0.5, zeroPoint: 127};
    let type0_length = product(type0.dimensions);
    let type25 = {type: nn.TENSOR_FLOAT32, dimensions: [10]};
    let type25_length = product(type25.dimensions);

    let input0 = operandIndex++;
    model.addOperand(type0);
    let output0 = operandIndex++;
    model.addOperand(type25);

    model.addOperation(nn.DEQUANTIZE, [input0], [output0]);

    model.identifyInputsAndOutputs([input0], [output0]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input0_input = new Uint8Array(input0_value);
    execution.setInput(0, input0_input);
    let output0_output = new Float32Array(type25_length);
    execution.setOutput(0, output0_output);

    await execution.startCompute();

    for (let i = 0; i < type25_length; ++i) {
      assert.isTrue(almostEqualCTS(output0_output[i], output0_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-4', async function() {
    // For 'Dequantize v1_2' example: examples_2d_quant8_asymm
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output01_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 5], scale: 0.5, zeroPoint: 127};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type3_length = product(type3.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type2);
    let output01 = operandIndex++;
    model.addOperand(type3);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Uint8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type3_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-5', async function() {
    // For 'Dequantize v1_2' example: examples_2d_quant8_asymm_relaxed
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output01_expect = [-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64];

    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 5], scale: 0.5, zeroPoint: 127};
    let type2_length = product(type2.dimensions);
    let type3 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type3_length = product(type3.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type2);
    let output01 = operandIndex++;
    model.addOperand(type3);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Uint8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type3_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type3_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-6', async function() {
    // For 'Dequantize v1_2' example: examples_2d_quant8_asymm_float16
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let input01_value = [0, 1, 2, 3, 4, 251, 252, 253, 254, 255];
    let output01_expect = [-63.5, -63.0, -62.5, -62.0, -61.5, 62.0, 62.5, 63.0, 63.5, 64.0];

    let type2 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [2, 5], scale: 0.5, zeroPoint: 127};
    let type2_length = product(type2.dimensions);
    let type26 = {type: nn.TENSOR_FLOAT32, dimensions: [2, 5]};
    let type26_length = product(type26.dimensions);

    let input01 = operandIndex++;
    model.addOperand(type2);
    let output01 = operandIndex++;
    model.addOperand(type26);

    model.addOperation(nn.DEQUANTIZE, [input01], [output01]);

    model.identifyInputsAndOutputs([input01], [output01]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let input01_input = new Uint8Array(input01_value);
    execution.setInput(0, input01_input);
    let output01_output = new Float32Array(type26_length);
    execution.setOutput(0, output01_output);

    await execution.startCompute();

    for (let i = 0; i < type26_length; ++i) {
      assert.isTrue(almostEqualCTS(output01_output[i], output01_expect[i]));
    }
  });

  it('check result for Dequantize v1_2 example-7', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input04, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-8', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input04, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-9', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input04, {channelDim: 0, scales: new Float32Array([2.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-10', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input05, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-11', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input05, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-12', async function() {
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
    model.setOperandSymmPerChannelQuantParams(input05, {channelDim: 1, scales: new Float32Array([2.0, 1.0, 0.5])});
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

  it('check result for Dequantize v1_2 example-13', async function() {
    // For 'Dequantize v1_2' example: examples
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [0, 32, 128, 255];
    let op2_expect = [0.0, 32.0, 128.0, 255.0];

    let type11 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 1.0, zeroPoint: 0};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_FLOAT32, dimensions: [1, 2, 2, 1]};
    let type12_length = product(type12.dimensions);

    let op1 = operandIndex++;
    model.addOperand(type11);
    let op2 = operandIndex++;
    model.addOperand(type12);

    model.addOperation(nn.DEQUANTIZE, [op1], [op2]);

    model.identifyInputsAndOutputs([op1], [op2]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op2_output = new Float32Array(type12_length);
    execution.setOutput(0, op2_output);

    await execution.startCompute();

    for (let i = 0; i < type12_length; ++i) {
      assert.isTrue(almostEqualCTS(op2_output[i], op2_expect[i]));
    }
  });
});
