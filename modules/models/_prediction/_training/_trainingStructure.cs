/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 18:46:31
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-07 09:11:31
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
using static modules.models.helpMethods.HelpMethods;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using modules.models.Base;
using static modules.models.Prediction.Utils;
using Tensorflow.Keras.Optimizers;

namespace modules.models.Prediction
{
    abstract class Model : Base.Model
    {
        TrainArgs _train_args;
        public Dictionary<string, Tensor> move_dict = new Dictionary<string, Tensor>();
        public Dictionary<string, Tensor> rotate_dict = new Dictionary<string, Tensor>();
        public Dictionary<string, Tensor> scale_dict = new Dictionary<string, Tensor>();

        public override TrainArgs args
        {
            get
            {
                return this._train_args;
            }
        }

        Structure _training_structure;
        public Structure training_structure
        {
            get
            {
                return this._training_structure;
            }
        }

        public Model(TrainArgs args, Structure training_structure) : base((BaseArgs)args)
        {
            this._train_args = args;
            this._training_structure = training_structure;
        }
    }


    class Structure : Base.Structure
    {
        TrainArgs _train_args;
        public new TrainArgs args
        {
            get
            {
                return this._train_args;
            }
        }

        public Structure(TrainArgs args) : base(args)
        {
            this._train_args = args;
        }

        public void run_commands()
        {
            // prepare training
            if (this.args.load == "null")
            {
                // this.model.load_weights("./test.tf");
                (this.model, _) = this.create_model();
                this.log_function("Training method is not support in current platform!");
                throw new KeyNotFoundException();
            }
            else
            {
                (this.model, _) = this.create_model();
                this.load_from_checkpoint(this.args.load);
                if (this.args.test_mode == "all")
                {
                    foreach (var dataset in new PredictionDatasetManager().ethucy_testsets)
                    {
                        var agents = load_dataset_files(this.args, dataset);
                        this.test(new Dictionary<string, object> { { "agents", agents }, { "dataset_name", dataset } });
                    }
                }
                else if (this.args.test_mode == "mix")
                {
                    var agents = new List<TrainAgentManager>();
                    string dataset = "";
                    foreach (var dataset_c in new PredictionDatasetManager().ethucy_testsets)
                    {
                        var agents_c = load_dataset_files(this.args, dataset_c);
                        agents.Concat(agents_c.ToList());
                        dataset.Concat(String.Format("{0}; ", dataset_c));
                    }
                    this.test(new Dictionary<string, object> { { "agents", agents }, { "dataset_name", dataset } });

                }
                else if (this.args.test_mode == "one")
                {
                    var agents = load_dataset_files(this.args, this.args.test_set);
                    this.test(new Dictionary<string, object> { { "agents", agents }, { "dataset_name", this.args.test_set } });
                }
            }
        }

        ///FUNCTION_NAME: create_model
        ///<summary>
        ///        Create models.
        ///        Please *rewrite* this when training new models.
        ///        
        ///</summary>
        ///<return name="model"> created model </return>
        new virtual public (Prediction.Model, Tensorflow.Keras.Optimizers.OptimizerV2) create_model()
        {
            Prediction.Model model = null;
            var opt = keras.optimizers.Adam(0.001f);
            return (model, opt);
        }

        ///FUNCTION_NAME: load_from_checkpoint
        ///<summary>
        ///        Load already trained models from `.h5` or `.tf` files according to args.
        ///
        ///</summary>
        ///<param name="model_path"> target dir where your model puts in </param>
        public virtual Model load_from_checkpoint(string model_path)
        {
            this.load_model(model_path);
            return null;
        }

        public virtual (List<Tensor> model_inputs, Tensor gt) get_inputs_from_agents(List<TrainAgentManager> input_agents)
        {
            return getInputs_onlyTraj(input_agents);
        }

        ///FUNCTION_NAME: load_dataset
        ///<summary>
        ///        Load training and val dataset.
        ///
        ///</summary>
        ///<return name="dataset_train"> train dataset, type = `tf.data.Dataset` </return>
        public override (Tensors, Tensors) load_dataset()
        {
            var dm = new TrainDataManager(this.args, prepare_type: "all");
            var agents_train = (List<TrainAgentManager>)dm.train_info["train_data"];
            var agents_test = (List<TrainAgentManager>)dm.train_info["test_data"];
            var train_number = dm.train_info["train_number"];
            var sample_time = dm.train_info["sample_time"];

            (Tensors train_model_inputs, Tensor train_labels) = this.get_inputs_from_agents(agents_train);
            (Tensors test_model_inputs, Tensor test_labels) = this.get_inputs_from_agents(agents_test);

            train_model_inputs.Add(train_labels);
            test_model_inputs.Add(test_labels);

            return (train_model_inputs, test_model_inputs);
        }

        ///FUNCTION_NAME: load_test_dataset
        ///<summary>
        ///        Load test dataset.
        ///
        ///</summary>
        public override Tensors load_test_dataset(Dictionary<string, object> kwargs = null)
        {
            var agents = (List<TrainAgentManager>)kwargs["agents"];
            (Tensors model_inputs, Tensor labels) = this.get_inputs_from_agents(agents);
            model_inputs.Add(labels);
            return model_inputs;
        }

        ///FUNCTION_NAME: load_forward_dataset
        ///<summary>
        ///        Load forward dataset.
        ///
        ///</summary>
        public override Tensors load_forward_dataset(Dictionary<string, object> kwargs = null)
        {
            dynamic agents = kwargs["model_inputs"];
            return getForwardDataset_onlyTraj(agents);
        }

        ///FUNCTION_NAME: loss
        ///<summary>
        ///        Train loss, using ADE by default.
        ///
        ///        
        ///</summary>
        ///<param name="outputs"> model's outputs </param>
        ///<param name="labels"> groundtruth labels </param>
        ///<param name="loss_name_list"> a list of name of used loss functions </param>
        ///<return name="loss"> sum of all single loss functions </return>
        public override (Tensor, Dictionary<string, Tensor>) loss(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null)
        {
            var loss = Loss.ADE(outputs[0], labels);
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("ADE", loss);
            return (loss, loss_dict);
        }

        ///FUNCTION_NAME: loss_eval
        ///<summary>
        ///        Eval loss, using [ADE, FDE] by default.
        ///
        ///        
        ///</summary>
        ///<param name="outputs"> model's outputs </param>
        ///<param name="labels"> groundtruth labels </param>
        ///<param name="loss_name_list"> a list of name of used loss functions </param>
        ///<return name="loss"> sum of all single loss functions </return>
        public override (Tensor, Dictionary<string, Tensor>) loss_eval(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null)
        {
            var loss_ade = Loss.ADE(outputs[0], labels);
            var loss_fde = Loss.FDE(outputs[0], labels);
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("ADE", loss_ade);
            loss_dict.Add("FDE", loss_fde);
            return (loss_ade, loss_dict);
        }

        public override void print_dataset_info()
        {
            this._print_info(title: "dataset options", new Dictionary<string, object> {
                {"title", "dataset options"},
                {"rotate_times", this.args.rotate},
                {"add_noise", this.args.add_noise},
            });
        }

        public override void print_training_info()
        {
            this._print_info(title: "training options");
        }

        public override void print_test_result_info(Dictionary<string, Tensor> loss_dict, Dictionary<string, object> kwargs = null)
        {
            dynamic dataset = kwargs["dataset_name"];
            var print_args = new Dictionary<string, object> { { "dataset", dataset } };
            foreach (var key in loss_dict.Keys)
            {
                print_args.Add(key, loss_dict[key]);
            }
            this._print_info(title: "test results", print_args);
        }

        public override void write_test_results(List<Tensor> model_outputs = null, Dictionary<string, object> kwargs = null)
        {
            dynamic agents = kwargs["agents"];
            dynamic testset_name = kwargs["dataset_name"];

            // TODO
        }
    }
}