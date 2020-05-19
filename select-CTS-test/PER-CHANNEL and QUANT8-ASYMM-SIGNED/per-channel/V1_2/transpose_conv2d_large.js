// Generated file (from: transpose_conv2d_large.mod.py). Do not edit
describe('CTS', function() {
  const assert = chai.assert;
  const nn = navigator.ml.getNeuralNetworkContext();

  it('check result for Transpose conv2d large example-1', async function() {
    // For 'Transpose conv2d large' example: examples_channelQuant8
    let model = await nn.createModel(options);
    let operandIndex = 0;

    let op1_value = [104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104, 104];

    let type11 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [25, 1, 1, 1], scale: 0.25, zeroPoint: 100};
    let type11_length = product(type11.dimensions);
    let type12 = {type: nn.TENSOR_QUANT8_SYMM_PER_CHANNEL, dimensions: [16, 1, 1, 1]};
    let type12_length = product(type12.dimensions);
    let type13 = {type: nn.TENSOR_INT32, dimensions: [16], scale: 0.0, zeroPoint: 0};
    let type13_length = product(type13.dimensions);
    let type14 = {type: nn.TENSOR_QUANT8_ASYMM, dimensions: [25, 32, 32, 16], scale: 0.5, zeroPoint: 80};
    let type14_length = product(type14.dimensions);
    let type4 = {type: nn.TENSOR_INT32, dimensions: [4]};
    let type4_length = product(type4.dimensions);
    let type5 = {type: nn.INT32};

    let op1 = operandIndex++;
    model.addOperand(type11);
    let op2 = operandIndex++;
    model.addOperand(type12);
    model.setOperandSymmPerChannelQuantParams(operandIndex++, {channelDim: 0, scales: new Float32Array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])});
    let op3 = operandIndex++;
    model.addOperand(type13);
    let shape = operandIndex++;
    model.addOperand(type4);
    let param = operandIndex++;
    model.addOperand(type5);
    let param1 = operandIndex++;
    model.addOperand(type5);
    let param2 = operandIndex++;
    model.addOperand(type5);
    let act = operandIndex++;
    model.addOperand(type5);
    let op4 = operandIndex++;
    model.addOperand(type14);

    model.setOperandValue(op2, new Int8Array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]));
    model.setOperandValue(op3, new Int32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    model.setOperandValue(shape, new Int32Array([25, 32, 32, 16]));
    model.setOperandValue(param, new Int32Array([1]));
    model.setOperandValue(param1, new Int32Array([32]));
    model.setOperandValue(param2, new Int32Array([32]));
    model.setOperandValue(act, new Int32Array([0]));
    model.addOperation(nn.TRANSPOSE_CONV_2D, [op1, op2, op3, shape, param, param1, param2, act], [op4]);

    model.identifyInputsAndOutputs([op1], [op4]);
    await model.finish();

    let compilation = await model.createCompilation();
    compilation.setPreference(getPreferenceCode(options.prefer));
    await compilation.finish();

    let execution = await compilation.createExecution();

    let op1_input = new Uint8Array(op1_value);
    execution.setInput(0, op1_input);
    let op4_output = new Uint8Array(type14_length);
    execution.setOutput(0, op4_output);

    await execution.startCompute();

    for (let i = 0; i < type14_length; ++i) {
      assert.isTrue(almostEqualCTS(op4_output[i], op4_expect[i]));
    }
  });
});