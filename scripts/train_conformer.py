
modelName = 'Log-PatchMask-Conformer-no-causal'

args = {}
args['model'] = 'Conformer'
args['outputDir'] = 'logs/speech_logs/' + modelName
args['datasetPath'] = 'datasets/ptDecoder_ctc_log_transform'
args['load_weights'] = True

args['load_weights_path'] = 'logs/speech_logs/' + modelName + '/modelWeights'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
args['lrStart'] = 1e-5
args['lrEnd'] = 1e-6
args['nUnits'] = 256
args['nhead'] = 8
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 1.0
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 5
args['kernelLen'] = 5
args['l2_decay'] = 1e-5
args['temporalMaxLen'] = 0
args['temporalMaskProb'] = 0
args['temporalNumMask'] = 4
args['patchMaskRatio'] = 0.075
args['patchNumMask'] = 20
args['causal'] = False

from src.neural_decoder.neural_decoder_trainer import trainModel
print(args)
trainModel(args)