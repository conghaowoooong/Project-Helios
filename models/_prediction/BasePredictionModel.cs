/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 18:46:31
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-29 00:58:14
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
using static models.Prediction.Utils;
using models.Managers.ArgManagers;
using models.Managers.AgentManagers;
using models.Managers.TrainManagers;
using models.Managers.DatasetManagers;

namespace models.Prediction
{
    class BasePredictionModel : TrainingStructure{
        public BasePredictionModel(TrainArgsManager args):base(args){
            
        }

        public void run_commands(){
            // prepare training
            if (this.args.load == "null")
            {
                this.log_function("Training method is not support in current platform!");
                throw new KeyNotFoundException();
            }
            else
            {
                this.model = this.load_from_checkpoint(this.args.load);
                if (this.args.test_mode == "all")
                {
                    foreach (var dataset in new PredictionDatasetManager().ethucy_testsets){
                        var agents = load_dataset_files(this.args, dataset);
                        this.test(new Dictionary<string, object> {{"agents", agents}, {"dataset_name", dataset}});
                    }
                } else if (this.args.test_mode == "mix") {
                    var agents = new List<TrainAgentManager>();
                    string dataset = "";
                    foreach (var dataset_c in new PredictionDatasetManager().ethucy_testsets){
                        var agents_c = load_dataset_files(this.args, dataset_c);
                        agents.concat(agents_c).ToList();
                        dataset.Concat(String.Format("{0}; ", dataset_c));
                    }
                    this.test(new Dictionary<string, object> {{"agents", agents}, {"dataset_name", dataset}});
                    
                } else if (this.args.test_mode == "one") {
                    var agents = load_dataset_files(this.args, this.args.test_set);
                    this.test(new Dictionary<string, object> {{"agents", agents}, {"dataset_name", this.args.test_set}});
                }
            }
        }

        public virtual Model load_from_checkpoint(string model_path){
            this.load_model(model_path);
            return this.model;
        }

        (List<Tensor> model_inputs, Tensor gt) prepare_model_inputs_all(List<TrainAgentManager> input_agents){
            return getInputs_onlyTraj(input_agents);
        }

        public override (IDatasetV2, IDatasetV2) load_dataset(){
            var dm = new TrainDataManager(this.args, prepare_type:"all");
            var agents_train = (List<TrainAgentManager>)dm.train_info["train_data"];
            var agents_test = (List<TrainAgentManager>)dm.train_info["test_data"];
            var train_number = dm.train_info["train_number"];
            var sample_time = dm.train_info["sample_time"];

            (Tensors train_model_inputs, Tensor train_labels) = this.prepare_model_inputs_all(agents_train);
            (Tensors test_model_inputs, Tensor test_labels) = this.prepare_model_inputs_all(agents_test);

            var dataset_train = tf.data.Dataset.from_tensor_slices(train_model_inputs, train_labels);
            dataset_train = dataset_train.shuffle(len(dataset_train), reshuffle_each_iteration:true);
            var dataset_test = tf.data.Dataset.from_tensor_slices(test_model_inputs, test_labels);

            return (dataset_train, dataset_test);
        }

        public override IDatasetV2 load_test_dataset(Dictionary<string, object> kwargs = null){
            var agents = (List<TrainAgentManager>)kwargs["agents"];
            (Tensors model_inputs, Tensor labels) = this.prepare_model_inputs_all(agents);
            var dataset_test = tf.data.Dataset.from_tensor_slices(model_inputs, labels);
            return dataset_test;
        }

        public override IDatasetV2 load_forward_dataset(Dictionary<string, object> kwargs = null){
            dynamic agents = kwargs["model_inputs"];
            return getForwardDataset_onlyTraj(agents);
        }

        public override (Tensor, Dictionary<string, Tensor>) loss(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null){
            var loss = Loss.ADE(outputs[0], labels);
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("ADE", loss);
            return (loss, loss_dict);
        }

        public override (Tensor, Dictionary<string, Tensor>) loss_eval(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null){
            var loss_ade = Loss.ADE(outputs[0], labels);
            var loss_fde = Loss.FDE(outputs[0], labels);
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("ADE", loss_ade);
            loss_dict.Add("FDE", loss_fde);
            return (loss_ade, loss_dict);
        }

        public override void print_dataset_info(){
            this._print_info(title:"dataset options", new Dictionary<string, object> {
                {"title", "dataset options"},
                {"rotate_times", this.args.rotate},
                {"add_noise", this.args.add_noise},
            });
        }

        public override void print_training_info(){
            this._print_info(title:"training options");
        }

        public override void print_test_result_info(Dictionary<string, Tensor> loss_dict, Dictionary<string, object> kwargs = null){
            dynamic dataset = kwargs["dataset_name"];
            var print_args = new Dictionary<string, object> {{"dataset", dataset}};
            foreach (var key in loss_dict.Keys){
                print_args.Add(key, loss_dict[key]);
            }
            this._print_info(title:"test results", print_args);
        }

        public override void write_test_results(List<Tensor> model_outputs = null, Dictionary<string, object> kwargs = null){
            dynamic agents = kwargs["agents"];
            dynamic testset_name = kwargs["dataset_name"];

            
        }
    }
}