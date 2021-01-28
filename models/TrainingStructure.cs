/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 14:33:11
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 12:57:34
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using NumSharp;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Engine;
using static models.HelpMethods;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using models.Managers;
using models.Managers.ArgManagers;

namespace models
{
    class TrainingStructure{
        public TrainArgsManager args;
        public Model model;
        Tensorflow.Keras.Optimizers.OptimizerV2 opt;
        
        public void log_function(string str){
            Console.WriteLine(str);
        }

        public TrainingStructure(TrainArgsManager args){
            this.args = args;
        }

        void gpu_config(){
            
        }

        TrainArgsManager load_args(){
            return this.args;
        }

        public void load_model(string model_path){
            (this.model, this.opt) = this.create_model();
            this.model.load_weights(model_path);
        }

        void save_model(string save_path){
            this.model.save(save_path);
        }

        (Model, Tensorflow.Keras.Optimizers.OptimizerV2) create_model(){
            Model model = null;
            var opt = keras.optimizers.Adam(0.001f);
            return (model, opt);
        }

        Tensors pre_process(Tensors model_inputs){
            return model_inputs;
        }

        Tensors post_process(Tensors model_outputs, Tensors model_inputs = null){
            return model_outputs;
        }

        (Tensor, Dictionary<string, Tensor>) loss(Tensor outputs, Tensor labels, Tensors model_inputs = null){
            var loss = tf.reduce_mean(tf.square(outputs - labels));
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("L2", loss);
            return (loss, loss_dict);
        }

        (Tensor, Dictionary<string, Tensor>) loss_eval(Tensor outputs, Tensor labels, string mode = "test"){
            return this.loss(outputs, labels);
        }

        Tensor _forward_(Tensors model_inputs, string mode = "test"){
            bool pre_process, post_process;
            switch (mode)
            {
                case "test":
                    pre_process = true;
                    post_process = true;
                    break;
                default:
                    pre_process = true;
                    post_process = false;
                    break;
            }
            
            if (pre_process)
            {
                model_inputs = this.pre_process(model_inputs);
            }
            var outputs = this.model.Apply(model_inputs);
            if (post_process){
                outputs = this.post_process(outputs, model_inputs:model_inputs);
            }
            
            return outputs;
        }

        (Tensors, Tensor, Dictionary<string, Tensor>) val_during_training(Tensors model_inputs, Tensor gt){
            var model_outputs = this._forward_(model_inputs, mode:"train");
            (var loss_eval, var loss_dict) = this.loss_eval(model_outputs, gt, mode:"val");
            return (model_outputs, loss_eval, loss_dict);
        }

        (Tensor, Dictionary<string, Tensor>, Tensor) gradient_operations(Tensors model_inputs, Tensor gt, Tensor loss_move_average){
            using var tape = tf.GradientTape();
            var model_outputs = this._forward_(model_inputs, mode:"train");
            (var loss, var loss_dict) = this.loss(model_outputs, gt, model_inputs:model_inputs);
            loss_move_average = 0.7 * loss + 0.3 * loss_move_average;

            var grads = tape.gradient(loss_move_average, this.model.trainable_variables);
            this.opt.apply_gradients(zip(grads, this.model.trainable_variables.Select(x => x as ResourceVariable)));

            return (loss, loss_dict, loss_move_average);
        }

        public virtual (DatasetV2 dataset_train, DatasetV2 dataset_val) load_dataset(){
            DatasetV2 dataset_train = null;
            DatasetV2 dataset_val = null;
            return (dataset_train, dataset_val);
        }

        DatasetV2 load_test_dataset(){
            DatasetV2 dataset_test = null;
            return dataset_test;
        }

        DatasetV2 load_forward_dataset(Tensors model_inputs = null){
            DatasetV2 dataset_forward = null;
            return dataset_forward;
        }

        void print_dataset_info(){
            this._print_info(new Dictionary<string, Tensor>(), title:"dataset options");
        }

        void print_training_info(){
            this._print_info(new Dictionary<string, Tensor>(), title:"training options");
        }

        void print_test_result_info(Dictionary<string, Tensor> loss_dict){
            this._print_info(new Dictionary<string, Tensor>(), title:"test results");
        }

        void print_training_done_info(){
            this.log_function("Training done.");
            this.log_function(String.Format("Tensorboard training log file is saved at `{0}`", this.args.log_dir));
            this.log_function(String.Format("To open this log file, please use `tensorboard --logdir {0}`", this.args.log_dir));
        }

        void _print_info(Dictionary<string, Tensor> dict, string title="null"){
            this.log_function(String.Format("-----------------{0}-----------------", title));
            foreach (var key in dict.Keys)
            {
                this.log_function(String.Format("Arg '{0}' is '{1}'", key, dict[key].numpy()));
            }
            this.log_function("\n");
        }

        void train(){
            this.log_function("Training method is not support in current platform!");
            throw new KeyNotFoundException();
        }

        (Tensors model_outputs, Tensor loss, Dictionary<string, Tensor> loss_dict) test_one_step(Tensors model_inputs, Tensor gt){
            var model_outputs = this._forward_(model_inputs, mode:"test");
            (var loss, var loss_dict) = this.loss_eval(model_outputs, gt, mode:"test");
            return (model_outputs, loss, loss_dict);
        }

        void write_test_results(List<Tensor> model_outputs = null){

        }

        void test(){
            // load dataset
            var dataset_test = this.load_test_dataset();

            // start test
            var time_bar = dataset_test.batch(this.args.batch_size);

            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();
            Dictionary<string, List<Tensor>> loss_dict_all = new Dictionary<string, List<Tensor>>();

            foreach (var test_data in time_bar){
                (var model_outputs, var loss, var loss_dict) = this.test_one_step(test_data.Item1, test_data.Item2);

                foreach (var index in range(len(model_outputs))){
                    if (len(model_outputs_all) < index + 1){
                        model_outputs_all.append(new List<Tensor>());
                    }
                    model_outputs_all[index].append(model_outputs[index]);
                }

                foreach (var key in loss_dict.Keys){
                    loss_dict_all[key].append(loss_dict[key]);
                }
            }

            List<Tensor> tensor_outputs = new List<Tensor>();
            foreach (var index in range(len(model_outputs_all))){
                tensor_outputs[index] = tf.concat(model_outputs_all[index], axis:0);
            }

            Dictionary<string, Tensor> loss_dict_outputs = new Dictionary<string, Tensor>();
            foreach (var key in loss_dict_all.Keys){
                loss_dict_outputs[key] = tf.reduce_mean(tf.stack(loss_dict_all[key])).numpy();
            }

            // write test results
            this.print_test_result_info(loss_dict_outputs);
            this.write_test_results(tensor_outputs);
        }

        List<Tensor> forward(DatasetV2 dataset){
            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();
            
            foreach (var model_inputs in dataset.batch(this.args.batch_size)){
                var model_outputs = this._forward_(model_inputs.Item1, mode:"test");

                foreach (var index in range(len(model_outputs))){
                    if (len(model_outputs_all) < index + 1){
                        model_outputs_all.append(new List<Tensor>());
                    }
                    model_outputs_all[index].append(model_outputs[index]);
                }
            }

            List<Tensor> tensor_outputs = new List<Tensor>();
            foreach (var index in range(len(model_outputs_all))){
                tensor_outputs[index] = tf.concat(model_outputs_all[index], axis:0);
            }

            return tensor_outputs;
        }

        List<NDArray> call(Tensors model_inputs){
            var test_dataset = this.load_forward_dataset(model_inputs:model_inputs);
            var tensor_outputs = this.forward(test_dataset);

            List<NDArray> results = new List<NDArray>();
            foreach (var index in range(len(tensor_outputs))){
                results[index] = tensor_outputs[index].numpy();
            }
            return results;
        }
    }
}