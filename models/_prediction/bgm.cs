/*
 * @Author: Conghao Wong
 * @Date: 2021-01-29 02:02:35
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 11:12:49
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static models.Prediction.Utils;

using static models.HelpMethods;
using models.Managers.ArgManagers;
using models.Managers.AgentManagers;
using models.Managers.TrainManagers;
using models.Managers.DatasetManagers;
using Tensorflow.Keras.Optimizers;

namespace models.Prediction
{
    class NewBGM : BasePredictionModel
    {
        private Tensor start_point;

        public NewBGM(TrainArgsManager args) : base(args)
        {

        }

        public override Tensors pre_process(Tensors model_inputs, Dictionary<string, object> kwargs = null)
        {
            Tensor positions = model_inputs[0];
            Tensor start_point = tf.reshape(tf.transpose(positions, (1, 0, 2))[-1], (-1, 1, 2));
            Tensor position_n = positions - start_point;
            this.start_point = start_point;
            
            return new Tensor[] {position_n, model_inputs[1], model_inputs[2]};
        }

        public override Tensors post_process(Tensors model_outputs, Tensors model_inputs = null)
        {
            Tensor output_positions = model_outputs;
            Tensor original_position = output_positions + this.start_point;

            return new Tensor[] {original_position};
        }

        public override (Model, OptimizerV2) create_model()
        {
            var positions = keras.layers.Input(shape: (this.args.obs_frames, 2));
            var maps = keras.layers.Input(shape: (100, 100));
            var paras = keras.layers.Input(shape: (2, 2));
            
            // sequence feature;
            // var positions_embadding_lstm = keras.layers.Dense(64, activation: "Tanh", input_shape : (this.args.obs_frames, 2)).Apply(positions);
            var flatten = keras.layers.Flatten().Apply(positions);
            // var traj_feature = keras.layers.LSTM(64, return_sequences: true).Apply(positions);
            // var feature_flatten = keras.layers.Flatten().Apply(traj_feature);
            var sequence_feature = keras.layers.Dense(this.args.pred_frames * 32, activation: "Tanh").Apply(flatten);

            // context feature;
            var maps_r = keras.layers.Reshape((100, 100, 1)).Apply(maps);
            var average_pooling = keras.layers.MaxPooling2D((3, 3)).Apply(maps_r);
            var cnn1 = keras.layers.Conv2D(32, (11, 11), activation: "Relu").Apply(average_pooling);
            var cnn2 = keras.layers.Conv2D(32, (3, 3), activation: "Relu").Apply(cnn1);
            var pooling2 = keras.layers.MaxPooling2D((3, 3)).Apply(cnn2);
            var cnn3 = keras.layers.Conv2D(12, (3, 3), activation: "Relu").Apply(pooling2);
            var flatten1 = keras.layers.Flatten().Apply(cnn3);
            var context_feature = keras.layers.Dense(this.args.pred_frames * 32, activation: "Tanh").Apply(flatten1);

            // joint feature;
            var concat_feature = keras.layers.Concatenate().Apply(new Tensor[] { sequence_feature, context_feature });
            var feature_fc = keras.layers.Dense(this.args.pred_frames * 64, activation: "Relu").Apply(concat_feature);
            var feature_fc2 = keras.layers.Dense(this.args.pred_frames * 2).Apply(feature_fc);
            var feature_reshape = keras.layers.Reshape(( this.args.pred_frames, 2)).Apply(feature_fc2);
            
            Tensors model_inputs = new Tensor[] { positions, maps, paras };
            this.model.load_weights("./test.tf")
            var lstm = keras.Model(model_inputs[0], feature_reshape);
            var lstm_optimizer = keras.optimizers.Adam(learning_rate: this.args.lr);

            lstm.summary();
            return (lstm, lstm_optimizer);
        }
    }
}