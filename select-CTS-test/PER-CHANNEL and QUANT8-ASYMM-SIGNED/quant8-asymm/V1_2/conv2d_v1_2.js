// Generated file (from: conv2d_v1_2.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Conv2d v1_2 example-1', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0625, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type33);
    let op3 = operandIndex++;
    model.addOperand(type34);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Uint8Array([2, 2, 2, 2]));
    model.setOperandValue(op3, new Int32Array([0]));
    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-2', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [2, 2, 2, 2, 1, 2, 2, 2, 2];
    let op2_value = [2, 2, 2, 2];
    let op3_value = [0];
    let op4_expect = [7, 7, 7, 7];

    let type32 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 0};
    let type32_length = product(type32.dimensions);
    let type33 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 2, 1], scale: 0.125, zeroPoint: 0};
    let type33_length = product(type33.dimensions);
    let type34 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.0625, zeroPoint: 0};
    let type34_length = product(type34.dimensions);
    let type4 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type32);
    let op2 = operandIndex++;
    model.addOperand(type33);
    let op3 = operandIndex++;
    model.addOperand(type34);
    let param = operandIndex++;
    model.addOperand(type4);
    let param1 = operandIndex++;
    model.addOperand(type4);
    let param2 = operandIndex++;
    model.addOperand(type4);
    let param3 = operandIndex++;
    model.addOperand(type4);
    let param4 = operandIndex++;
    model.addOperand(type4);
    let param5 = operandIndex++;
    model.addOperand(type4);
    let param6 = operandIndex++;
    model.addOperand(type4);
    let op4 = operandIndex++;
    model.addOperand(type33);

    model.setOperandValue(op2, new Uint8Array(op2_value));
    model.setOperandValue(op3, new Int32Array(op3_value));

    model.setOperandValue(param, new Int32Array([0]));
    model.setOperandValue(param1, new Int32Array([0]));
    model.setOperandValue(param2, new Int32Array([0]));
    model.setOperandValue(param3, new Int32Array([0]));
    model.setOperandValue(param4, new Int32Array([1]));
    model.setOperandValue(param5, new Int32Array([1]));
    model.setOperandValue(param6, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op1, op2, op3, param, param1, param2, param3, param4, param5, param6], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type33_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type33_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-3', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type47 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 127};
    let type47_length = product(type47.dimensions);
    let type48 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.25, zeroPoint: 0};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type47);
    let op31 = operandIndex++;
    model.addOperand(type48);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Uint8Array([129, 135, 141, 131, 137, 143, 133, 139, 145]));
    model.setOperandValue(op31, new Int32Array([-800]));
    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-4', async function() {
    // For 'Conv2d v1_2' example: examples_nhwc_weight_as_input_quant8_2
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op11_value = [129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151];
    let op21_value = [129, 135, 141, 131, 137, 143, 133, 139, 145];
    let op31_value = [-800];
    let op41_expect = [50, 50, 50, 50, 85, 162, 207, 50, 50, 84, 111, 50];

    let type4 = {type: nn.INT32};
    let type46 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 0.5, zeroPoint: 127};
    let type46_length = product(type46.dimensions);
    let type47 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 3, 1], scale: 0.5, zeroPoint: 127};
    let type47_length = product(type47.dimensions);
    let type48 = {type: nn.TENSOR_INT32, dimensions: [1], scale: 0.25, zeroPoint: 0};
    let type48_length = product(type48.dimensions);
    let type49 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 3, 4, 1], scale: 1.0, zeroPoint: 50};
    let type49_length = product(type49.dimensions);

    let op11 = operandIndex++;
    model.addOperand(type46);
    let op21 = operandIndex++;
    model.addOperand(type47);
    let op31 = operandIndex++;
    model.addOperand(type48);
    let param7 = operandIndex++;
    model.addOperand(type4);
    let param8 = operandIndex++;
    model.addOperand(type4);
    let param9 = operandIndex++;
    model.addOperand(type4);
    let param10 = operandIndex++;
    model.addOperand(type4);
    let op41 = operandIndex++;
    model.addOperand(type49);

    model.setOperandValue(op21, new Uint8Array(op21_value));
    model.setOperandValue(op31, new Int32Array(op31_value));

    model.setOperandValue(param7, new Int32Array([1]));
    model.setOperandValue(param8, new Int32Array([1]));
    model.setOperandValue(param9, new Int32Array([1]));
    model.setOperandValue(param10, new Int32Array([1]));
    model.addOperation(nn.CONV_2D, [op11, op21, op31, param7, param8, param9, param10], [op41]);

    model.identifyInputsAndOutputs([op11], [op41]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op11_input = new Uint8Array(op11_value);
    execution.setInput(0, op11_input);
    let op41_output = new Uint8Array(type49_length);
    execution.setOutput(0, op41_output);

    await execution.startCompute();

    for (let i = 0; i < type49_length; ++i) {
      assert.isTrue(almostEqualCTS(op41_output[i], op41_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-5', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type58 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type58_length = product(type58.dimensions);
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.25, zeroPoint: 0};
    let type59_length = product(type59.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type58);
    let op32 = operandIndex++;
    model.addOperand(type59);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    model.setOperandValue(op32, new Int32Array([0, 0, 0]));
    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-6', async function() {
    // For 'Conv2d v1_2' example: examples_channel_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op12_value = [10, 10, 10];
    let op22_value = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let op32_value = [0, 0, 0];
    let op42_expect = [30, 75, 120];

    let type4 = {type: nn.INT32};
    let type57 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type57_length = product(type57.dimensions);
    let type58 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 0};
    let type58_length = product(type58.dimensions);
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.25, zeroPoint: 0};
    let type59_length = product(type59.dimensions);

    let op12 = operandIndex++;
    model.addOperand(type57);
    let op22 = operandIndex++;
    model.addOperand(type58);
    let op32 = operandIndex++;
    model.addOperand(type59);
    let param11 = operandIndex++;
    model.addOperand(type4);
    let param12 = operandIndex++;
    model.addOperand(type4);
    let param13 = operandIndex++;
    model.addOperand(type4);
    let param14 = operandIndex++;
    model.addOperand(type4);
    let param15 = operandIndex++;
    model.addOperand(type4);
    let param16 = operandIndex++;
    model.addOperand(type4);
    let param17 = operandIndex++;
    model.addOperand(type4);
    let op42 = operandIndex++;
    model.addOperand(type57);

    model.setOperandValue(op22, new Uint8Array(op22_value));
    model.setOperandValue(op32, new Int32Array(op32_value));

    model.setOperandValue(param11, new Int32Array([0]));
    model.setOperandValue(param12, new Int32Array([0]));
    model.setOperandValue(param13, new Int32Array([0]));
    model.setOperandValue(param14, new Int32Array([0]));
    model.setOperandValue(param15, new Int32Array([1]));
    model.setOperandValue(param16, new Int32Array([1]));
    model.setOperandValue(param17, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op12, op22, op32, param11, param12, param13, param14, param15, param16, param17], [op42]);

    model.identifyInputsAndOutputs([op12], [op42]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op12_input = new Uint8Array(op12_value);
    execution.setInput(0, op12_input);
    let op42_output = new Uint8Array(type57_length);
    execution.setOutput(0, op42_output);

    await execution.startCompute();

    for (let i = 0; i < type57_length; ++i) {
      assert.isTrue(almostEqualCTS(op42_output[i], op42_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-7', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.25, zeroPoint: 0};
    let type59_length = product(type59.dimensions);
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type69 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 128};
    let type69_length = product(type69.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type69);
    let op33 = operandIndex++;
    model.addOperand(type59);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Uint8Array([130, 136, 142, 132, 138, 144, 134, 140, 146]));
    model.setOperandValue(op33, new Int32Array([0, 0, 0]));
    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });

  it('check result for Conv2d v1_2 example-8', async function() {
    // For 'Conv2d v1_2' example: examples_large_nhwc_weight_as_input_quant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op13_value = [130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160, 162, 164];
    let op23_value = [130, 136, 142, 132, 138, 144, 134, 140, 146];
    let op33_value = [0, 0, 0];
    let op43_expect = [15, 18, 21, 33, 40, 48, 51, 63, 75, 69, 86, 102, 87, 108, 129, 105, 130, 156];

    let type4 = {type: nn.INT32};
    let type59 = {type: nn.TENSOR_INT32, dimensions: [3], scale: 0.25, zeroPoint: 0};
    let type59_length = product(type59.dimensions);
    let type68 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 0.5, zeroPoint: 128};
    let type68_length = product(type68.dimensions);
    let type69 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [3, 1, 1, 3], scale: 0.5, zeroPoint: 128};
    let type69_length = product(type69.dimensions);
    let type70 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [1, 2, 3, 3], scale: 2.0, zeroPoint: 0};
    let type70_length = product(type70.dimensions);

    let op13 = operandIndex++;
    model.addOperand(type68);
    let op23 = operandIndex++;
    model.addOperand(type69);
    let op33 = operandIndex++;
    model.addOperand(type59);
    let param18 = operandIndex++;
    model.addOperand(type4);
    let param19 = operandIndex++;
    model.addOperand(type4);
    let param20 = operandIndex++;
    model.addOperand(type4);
    let param21 = operandIndex++;
    model.addOperand(type4);
    let param22 = operandIndex++;
    model.addOperand(type4);
    let param23 = operandIndex++;
    model.addOperand(type4);
    let param24 = operandIndex++;
    model.addOperand(type4);
    let op43 = operandIndex++;
    model.addOperand(type70);

    model.setOperandValue(op23, new Uint8Array(op23_value));
    model.setOperandValue(op33, new Int32Array(op33_value));

    model.setOperandValue(param18, new Int32Array([0]));
    model.setOperandValue(param19, new Int32Array([0]));
    model.setOperandValue(param20, new Int32Array([0]));
    model.setOperandValue(param21, new Int32Array([0]));
    model.setOperandValue(param22, new Int32Array([1]));
    model.setOperandValue(param23, new Int32Array([1]));
    model.setOperandValue(param24, new Int32Array([0]));
    model.addOperation(nn.CONV_2D, [op13, op23, op33, param18, param19, param20, param21, param22, param23, param24], [op43]);

    model.identifyInputsAndOutputs([op13], [op43]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op13_input = new Uint8Array(op13_value);
    execution.setInput(0, op13_input);
    let op43_output = new Uint8Array(type70_length);
    execution.setOutput(0, op43_output);

    await execution.startCompute();

    for (let i = 0; i < type70_length; ++i) {
      assert.isTrue(almostEqualCTS(op43_output[i], op43_expect[i]));
    }
  });
});
