
modelName = 'speech_log_transformed_seps_finetuned'

args = {}
args['pretrainedWeights'] = 'logs/speech_logs/speech_log_transformed_samll_eps/modelWeights'
args['outputDir'] = 'logs/speech_logs/' + modelName
args['datasetPath'] = 'datasets/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 16
args['lrStart'] = 1e-5
args['lrEnd'] = 5e-7
args['nUnits'] = 1024
args['nBatch'] = 3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)