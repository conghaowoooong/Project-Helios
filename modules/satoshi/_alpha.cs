/*
 * @Author: Conghao Wong
 * @Date: 2021-04-06 10:14:06
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-07 14:32:26
 * @Description: file content
 */

using NumSharp;
using System.Collections.Generic;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Optimizers;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static modules.models.helpMethods.HelpMethods;

using Prediction = modules.models.Prediction;
using modules.models.Prediction;

namespace modules.satoshi
    {
    class SatoshiAlphaModel : Prediction.Model {
        int gcn_layer_count;
        int intention_count;
        
        SatoshiAlphaModel self {
            get {
                return this;
            }
        }

        // GCN layers
        public Tensorflow.Keras.Layers.Dense pos_embadding;
        public Tensorflow.Keras.Layers.Dense adj_dense;
        public Dictionary<string, Tensorflow.Keras.Layers.Dense> gcn_layers = new Dictionary<string, Tensorflow.Keras.Layers.Dense>();

        public Tensorflow.Keras.Layers.Dropout gcn_dropout;
        public Tensorflow.Keras.Layers.BatchNormalization gcn_bn;

        public Tensorflow.Keras.Layers.Dense adj_dense2;
        public Tensorflow.Keras.Layers.Dense gcn_transfer;

        // context feature
        public Tensorflow.Keras.Layers.Pooling2D average_pooling;
        public Tensorflow.Keras.Layers.Flatten flatten;
        public Tensorflow.Keras.Layers.Dense context_dense1;

        // decoder
        public Tensorflow.Keras.Layers.Concatenate concat;
        public Tensorflow.Keras.Layers.Dense decoder;
        
        public SatoshiAlphaModel(
            Prediction.TrainArgs args,
            Prediction.Structure training_structure=null,
            int gcn_layer_count=2,
            int intention_count=10
        ) : base(args, training_structure) {
            self.gcn_layer_count = gcn_layer_count;
            self.intention_count = intention_count;

            // GCN layers
            self.pos_embadding = keras.layers.Dense(64, activation : tf.nn.tanh);
            self.adj_dense = keras.layers.Dense(self.args.obs_frames, activation : tf.nn.tanh);

            foreach (var count in range(self.gcn_layer_count)){
                self.gcn_layers[string.Format("{0}", count)] = GraphConv_layer(64, activation : count < self.gcn_layer_count - 1 ? tf.nn.relu : tf.nn.tanh);
            }
            
            self.gcn_dropout = keras.layers.Dropout(self.args.dropout);
            self.gcn_bn = keras.layers.BatchNormalization();

            self.adj_dense2 = keras.layers.Dense(self.intention_count, activation : tf.nn.tanh);
            self.gcn_transfer = GraphConv_layer(64, tf.nn.tanh);

            // context feature
            self.average_pooling = keras.layers.MaxPooling2D((5, 5));
            self.flatten = keras.layers.Flatten();
            self.context_dense1 = keras.layers.Dense(self.intention_count * 64, activation : tf.nn.tanh);

            // decoder
            self.concat = keras.layers.Concatenate();
            self.decoder = keras.layers.Dense(2);  
        }

        public override void build()
        {
            // historical GCN -> historical feature ([batch, obs, 64])
            var inputs1 = keras.layers.Input((2));
            var positions_embadding = self.pos_embadding.Apply(inputs1);
            var adj_matrix = self.adj_dense.Apply(positions_embadding);
            
            var gcn_input = positions_embadding;
            var gcn_output = gcn_input;
            foreach (var repeat in range(self.gcn_layer_count)){
                gcn_output = GraphConv_func(gcn_input, adj_matrix, layer:self.gcn_layers[string.Format("{0}", repeat)], building:true);
                gcn_input = gcn_output;
            }
            
            var dropout = self.gcn_dropout.Apply(gcn_output, training:false);
            var historical_feature = self.gcn_bn.Apply(dropout);

            // context feature -> context feature ([batch, K, 64])
            var maps = keras.layers.Input((100, 100));
            var maps_r = tf.expand_dims(maps, -1);
            var average_pooling = self.average_pooling.Apply(maps_r);
            var flatten = self.flatten.Apply(average_pooling);
            var context_feature = self.context_dense1.Apply(flatten);
            context_feature = tf.reshape(context_feature, (-1, self.intention_count, 64));

            // transfer GCN
            var adj_matrix_transfer_T = self.adj_dense2.Apply(positions_embadding);   // shape = [batch, obs, pred]
            var adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, (0, 2, 1));   // shape = [batch, pred, obs]
            var future_feature = GraphConv_func(historical_feature, adj_matrix_transfer, layer:self.gcn_transfer, building:true);

            // decoder
            var concat_feature = keras.layers.Input((128));
            var predictions = self.decoder.Apply(concat_feature);
        }

        public override Tensors call(Tensors inputs, bool training = false, dynamic mask = null)
        {
            var positions = inputs[0];
            var maps = inputs[1];
            
            // historical GCN -> historical feature ([batch, obs, 64])
            var positions_embadding = batch_dense(positions, self.pos_embadding);
            var adj_matrix = batch_dense(positions_embadding, self.adj_dense);
            
            var gcn_input = positions_embadding;
            var gcn_output = gcn_input;
            foreach (var repeat in range(self.gcn_layer_count)){
                gcn_output = GraphConv_func(gcn_input, adj_matrix, layer:self.gcn_layers[string.Format("{0}", repeat)]);
                gcn_input = gcn_output;
            }
            
            var dropout = self.gcn_dropout.Apply(gcn_output, training:training);
            var historical_feature = self.gcn_bn.Apply(dropout);    // BUG <= BN layer有问题

            // context feature -> context feature ([batch, K, 64])
            var maps_r = tf.expand_dims(maps, -1);
            var average_pooling = self.average_pooling.Apply(maps_r);
            var flatten = self.flatten.Apply(average_pooling);
            var context_feature = self.context_dense1.Apply(flatten);
            context_feature = tf.reshape(context_feature, (-1, self.intention_count, 64));

            // transfer GCN
            var adj_matrix_transfer_T = batch_dense(positions_embadding, self.adj_dense2);   // shape = [batch, obs, pred]
            var adj_matrix_transfer = tf.transpose(adj_matrix_transfer_T, (0, 2, 1));   // shape = [batch, pred, obs]
            var future_feature = GraphConv_func(historical_feature, adj_matrix_transfer, layer:self.gcn_transfer);

            // decoder
            var concat_feature = self.concat.Apply((future_feature, context_feature));
            var predictions = batch_dense(concat_feature, self.decoder);

            return predictions;
        }

        public override Tensors pre_process(Tensors model_inputs, Dictionary<string, object> kwargs = null)
        {
            var trajs = model_inputs[0];
            (trajs, self.move_dict) = Prediction.Process.move(trajs);
            // trajs, self.rotate_dict = Prediction.Process.rotate(trajs)
            // trajs, self.scale_dict = Prediction.Process.scale(trajs)
            return Prediction.Process.update(trajs, model_inputs);
        }

        public override Tensors post_process(Tensors model_outputs, Tensors model_inputs = null)
        {
            var trajs = model_outputs[0];
            // trajs = Prediction.Process.scale_back(trajs, self.scale_dict);
            // trajs = Prediction.Process.rotate_back(trajs, self.rotate_dict);
            trajs = Prediction.Process.move_back(trajs, self.move_dict);
            return Prediction.Process.update(trajs, model_outputs);
        }

        public override Tensors post_process_test(Tensors model_outputs, Tensors model_inputs = null)
        {
            var current_positions = tf.transpose(model_inputs[0], (1, 0, 2))[-1];   // [batch, 2]
            var intentions = model_outputs[0];

            List<Tensor> final_predictions = new List<Tensor>();
            foreach (var pred in range(1, self.args.pred_frames+1)){
                var final_pred = (intentions - tf.expand_dims(current_positions, 1))* pred / self.args.pred_frames + tf.expand_dims(current_positions, 1);
                final_predictions.append(final_pred);
            }
            var final_predictions_ = tf.transpose(tf.stack(final_predictions), (1, 2, 0, 3));
        
            return Prediction.Process.update(final_predictions_, model_outputs);
        }
    }

    class SatoshiAlpha : Prediction.Structure {
        int gcn_layer_count;
        SatoshiAlpha self {
            get {
                return this;
            }
        }

        public SatoshiAlpha(
            Prediction.TrainArgs args
        ) : base (args) {
            self.gcn_layer_count = self.args.gcn_layers;
        }

        public override (Prediction.Model, OptimizerV2) create_model()
        {
            var model = new SatoshiAlphaModel(
                self.args, 
                gcn_layer_count:2, 
                intention_count:self.args.K_train, 
                training_structure:self
            );
            model.build();
            var opt = keras.optimizers.Adam(self.args.lr);
            return (model, opt);
        }

        public override (List<Tensor> model_inputs, Tensor gt) get_inputs_from_agents(List<TrainAgentManager> input_agents)
        {
            return Prediction.Utils.getInputs_TrajWithMap(input_agents);
        }

        public override Tensors load_forward_dataset(Dictionary<string, object> kwargs = null)
        {
            var model_inputs = (List<TrainAgentManager>)(kwargs["model_inputs"]);
            return Prediction.Utils.getForwardDataset_TrajWithMap(model_inputs);
        }
    }

}
