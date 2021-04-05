/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 14:33:11
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-06 00:24:15
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using NumSharp;
using System.Linq;
using Tensorflow;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace modules.models.Base
{
    class Model : Tensorflow.Keras.Engine.Model
    {
        BaseArgs _base_args;
        Structure _training_structure;

        public Model(
            BaseArgs Args,
            Structure training_structure = null
        ) : base(new Tensorflow.Keras.ArgsDefinition.ModelArgs())
        {
            this._base_args = Args;
            this._training_structure = training_structure;
        }

        public virtual BaseArgs args
        {
            get
            {
                return this._base_args;
            }
        }

        Structure training_structure
        {
            get
            {
                return this._training_structure;
            }
        }

        public virtual Tensors call(Tensors inputs, bool training = false, dynamic mask = null)
        {
            // TODO call in keras.Model
            return null;
        }

        ///<summary>
        ///        Run a forward implementation.
        ///
        ///</summary>
        ///<param name="model_inputs" > input tensor (or a list of tensors)</param>
        ///<param name="mode" > choose forward type, can be `'test'` or `'train'`</param>
        public Tensors forward(Tensors model_inputs, bool training = false)
        {
            var post_process = training ? false : true;

            var model_inputs_processed = this.pre_process(model_inputs);
            var outputs = this.call(model_inputs);
            if (true)
            {
                // FIXME Tensor -> List[Tensor]
                var outputs_list = new List<Tensor>();
                outputs_list.append(outputs);
                outputs = outputs_list;
            }
            var output_processed = this.post_process(outputs);

            if (post_process)
            {
                outputs = this.post_process_test(outputs, model_inputs: model_inputs);
            }

            return outputs;
        }

        public virtual Tensors pre_process(Tensors model_inputs, Dictionary<string, object> kwargs = null)
        {
            return model_inputs;
        }

        public virtual Tensors post_process(Tensors model_outputs, Tensors model_inputs = null)
        {
            return model_outputs;
        }

        public virtual Tensors post_process_test(Tensors model_outputs, Tensors model_inputs = null)
        {
            return model_outputs;
        }
    }


    class Structure
    {
        BaseArgs _base_args;
        Model _base_model;
        private Tensorflow.Keras.Optimizers.OptimizerV2 opt;

        public virtual BaseArgs args
        {
            get
            {
                return this._base_args;
            }
        }

        public virtual dynamic model
        {
            get
            {
                return this._base_model;
            }
            set
            {
                this._base_model = value;
            }
        }

        public Structure(BaseArgs args)
        {
            this._base_args = args;
            this._gpu_config();
            this.load_args();
        }

        public void log_function(string str)
        {
            Console.WriteLine(str);
        }


        void _gpu_config()
        {
            // TODO gpu config
        }

        ///FUNCTION_NAME: load_args
        ///<summary>
        ///        Load args (`Namespace`) from `load_path` into `this.__args`
        ///
        ///</summary>
        ///<param name="current_args" > default args</param>
        ///<param name="load_path" > path of new args to load</param>
        public virtual BaseArgs load_args()
        {
            // TODO load args
            // TODO arg type choose
            return this.args;
        }

        ///FUNCTION_NAME: load_model
        ///<summary>
        ///        Load already trained models from checkpoint files.
        ///
        ///</summary>
        ///<param name="model_path"> path of model </param>
        public Model load_model(string model_path)
        {
            (model, opt) = this.create_model();
            model.load_weights(model_path);
            return model;
        }

        ///FUNCTION_NAME: save_model
        ///<summary>
        ///        Save trained model to `save_path`.
        ///
        ///</summary>
        void save_model(string save_path)
        {
            // FIXME it should be `save_weights` rather than `save`
            this.model.save(save_path);
        }

        ///FUNCTION_NAME: create_model
        ///<summary>
        ///        Create models.
        ///        Please *rewrite* this when training new models.
        ///        
        ///</summary>
        ///<return name="model"> created model </return>
        public virtual (Model, Tensorflow.Keras.Optimizers.OptimizerV2) create_model()
        {
            Model model = null;
            var opt = keras.optimizers.Adam(0.001f);
            return (model, opt);
        }

        ///FUNCTION_NAME: model_forward
        ///<summary>
        ///        Entire forward process of this model.
        ///
        ///</summary>
        ///<param name="model_inputs"> a list (or tuple) of tensor to input to model(s) </param>
        ///<param name="mode"> forward type, canbe `'test'` or `'train'` </param>
        Tensors model_forward(Tensors model_inputs, string mode = "test", Dictionary<string, object> kwargs = null)
        {
            return this.model.forward(model_inputs, training: mode == "test" ? false : true);
        }

        ///FUNCTION_NAME: loss
        ///<summary>
        ///        Train loss, using L2 loss by default.
        ///
        ///        
        ///</summary>
        ///<param name="outputs"> model's outputs </param>
        ///<param name="labels"> groundtruth labels </param>
        ///<param name="loss_name_list"> a list of name of used loss functions </param>
        ///<return name="loss"> sum of all single loss functions </return>
        public virtual (Tensor, Dictionary<string, Tensor>) loss(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null)
        {
            var loss = tf.reduce_mean(tf.square(outputs[0] - labels));
            var loss_dict = new Dictionary<string, Tensor>();

            loss_dict.Add("L2", loss);
            return (loss, loss_dict);
        }

        ///FUNCTION_NAME: loss_eval
        ///<summary>
        ///        Eval loss, using L2 loss by default.
        ///
        ///        
        ///</summary>
        ///<param name="outputs"> model's outputs </param>
        ///<param name="labels"> groundtruth labels </param>
        ///<param name="loss_name_list"> a list of name of used loss functions </param>
        ///<return name="loss"> sum of all single loss functions </return>
        public virtual (Tensor, Dictionary<string, Tensor>) loss_eval(Tensors outputs, Tensor labels, Dictionary<string, object> kwargs = null)
        {
            return this.loss(outputs, labels);
        }

        /// <Summary> Run one step of forward and calculate loss. </Summary>
        /// <param name="model_inputs"> a list of tensor as model's inputs </param>
        (Tensors, Tensor, Dictionary<string, Tensor>) _run_one_step(Tensors model_inputs, Tensor gt)
        {
            var model_outputs = this.model_forward(model_inputs, mode: "train");
            (var loss_eval, var loss_dict) = this.loss_eval(model_outputs, gt, new Dictionary<string, object> { { "mode", "val" } });
            return (model_outputs, loss_eval, loss_dict);
        }

        ///FUNCTION_NAME: gradient_operations
        ///<summary>
        ///        Run gradient dencent once during training.
        ///
        ///
        ///</summary>
        ///<param name="model_inputs"> model inputs </param>
        ///<param name="gt"> :ground truth </param>
        ///<param name="loss_move_average"> Moving average loss </param>
        ///<return name="loss"> sum of all single loss functions </return>
        ///<return name="loss_dict"> a dict of all loss functions </return>
        (Tensor, Dictionary<string, Tensor>, Tensor) gradient_operations(Tensors model_inputs, Tensor gt, Tensor loss_move_average, Dictionary<string, object> kwargs = null)
        {
            using var tape = tf.GradientTape();
            var model_outputs = this.model_forward(model_inputs, mode: "train");
            (var loss, var loss_dict) = this.loss(model_outputs, gt, kwargs);
            loss_move_average = 0.7 * loss + 0.3 * loss_move_average;

            var grads = tape.gradient(loss_move_average, this.model.trainable_variables);
            this.opt.apply_gradients(zip(grads, ((Model)this.model).trainable_variables.Select(x => x as ResourceVariable)));

            return (loss, loss_dict, loss_move_average);
        }
        
        ///FUNCTION_NAME: load_dataset
        ///<summary>
        ///        Load training and val dataset.
        ///
        ///</summary>
        ///<return name="dataset_train"> train dataset, type = `tf.data.Dataset` </return>
        public virtual (IDatasetV2 dataset_train, IDatasetV2 dataset_val) load_dataset()
        {
            IDatasetV2 dataset_train = null;
            IDatasetV2 dataset_val = null;
            return (dataset_train, dataset_val);
        }

        ///FUNCTION_NAME: load_test_dataset
        ///<summary>
        ///        Load test dataset.
        ///
        ///</summary>
        public virtual IDatasetV2 load_test_dataset(Dictionary<string, object> kwargs = null)
        {
            IDatasetV2 dataset_test = null;
            return dataset_test;
        }
        
        ///FUNCTION_NAME: load_forward_dataset
        ///<summary>
        ///        Load forward dataset.
        ///
        ///</summary>
        public virtual IDatasetV2 load_forward_dataset(Dictionary<string, object> kwargs = null)
        {
            IDatasetV2 dataset_forward = null;
            return dataset_forward;
        }

        public virtual void print_dataset_info()
        {
            this._print_info(title: "dataset options");
        }

        public virtual void print_training_info()
        {
            this._print_info(title: "training options");
        }

        public virtual void print_test_result_info(Dictionary<string, Tensor> loss_dict, Dictionary<string, object> kwargs = null)
        {
            this._print_info(title: "test results");
        }

        void print_training_done_info(Dictionary<string, object> kwargs = null)
        {
            this.log_function("Training done.");
            this.log_function(String.Format("Tensorboard training log file is saved at `{0}`", this.args.log_dir));
            this.log_function(String.Format("To open this log file, please use `tensorboard --logdir {0}`", this.args.log_dir));
        }

        public void _print_info(string title = "null", Dictionary<string, object> kwargs = null)
        {
            this.log_function(String.Format("-----------------{0}-----------------", title));
            foreach (var key in kwargs.Keys)
            {
                dynamic value = kwargs[key];
                if (value.GetType().IsSubclassOf(typeof(Tensor)))
                {
                    this.log_function(String.Format("Arg '{0}' is '{1}'", key, value.numpy()));
                }
                else
                {
                    this.log_function(String.Format("Arg '{0}' is '{1}'", key, value));
                }
            }
            this.log_function("\n");
        }

        void train()
        {
            this.log_function("Training method is not support in current platform!");
            throw new KeyNotFoundException();
        }

        (Tensors model_outputs, Tensor loss, Dictionary<string, Tensor> loss_dict) test_one_step(Tensors model_inputs, Tensor gt)
        {
            var model_outputs = this.model_forward(model_inputs, mode: "test");
            (var loss, var loss_dict) = this.loss_eval(model_outputs, gt, new Dictionary<string, object> { { "mode", "test" } });
            return (model_outputs, loss, loss_dict);
        }

        public virtual void write_test_results(List<Tensor> model_outputs = null, Dictionary<string, object> kwargs = null)
        {

        }

        public void test(Dictionary<string, object> kwargs = null)
        {
            // load dataset
            var dataset_test = this.load_test_dataset(kwargs);

            // start test
            var time_bar = dataset_test.batch(this.args.batch_size);

            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();
            Dictionary<string, List<Tensor>> loss_dict_all = new Dictionary<string, List<Tensor>>();

            foreach (var test_data in time_bar)
            {
                (var model_outputs, var loss, var loss_dict) = this.test_one_step(test_data.Item1, test_data.Item2);

                int length;
                if (model_outputs.GetType() == typeof(Tensors))
                {
                    length = model_outputs.Length;
                }
                else
                {
                    length = 1;
                }
                foreach (var index in range(length))
                {
                    if (len(model_outputs_all) < index + 1)
                    {
                        model_outputs_all.append(new List<Tensor>());
                    }
                    model_outputs_all[index].append(model_outputs[index]);
                }

                foreach (var key in loss_dict.Keys)
                {
                    if (!(loss_dict_all.ContainsKey(key)))
                    {
                        loss_dict_all[key] = new List<Tensor>();
                    }
                    loss_dict_all[key].append(loss_dict[key]);
                }
            }

            List<Tensor> tensor_outputs = new List<Tensor>();
            foreach (var index in range(len(model_outputs_all)))
            {
                tensor_outputs.append(tf.concat(model_outputs_all[index], axis: 0));
            }

            Dictionary<string, Tensor> loss_dict_outputs = new Dictionary<string, Tensor>();
            foreach (var key in loss_dict_all.Keys)
            {
                loss_dict_outputs[key] = tf.reduce_mean(tf.stack(loss_dict_all[key].ToArray()));
            }

            // write test results
            this.print_test_result_info(loss_dict_outputs, kwargs);
            this.write_test_results(tensor_outputs, kwargs);
        }
        
        ///FUNCTION_NAME: forward
        ///<summary>
        ///
        ///</summary>
        ///Forward model on one dataset and return outputs. </return>
        ///<param name="dataset"> dataset to forward, type = `tf.data.Dataset` </param>
        ///<param name="return_numpy"> controls if returns `numpy.ndarray` or `tf.Tensor` </param> </returns> </return>
        ///<param name="return_numpy"> controls if returns `numpy.ndarray` or `tf.Tensor` </param> </returns> </return>
        ///<param name="return_numpy"> controls if returns `numpy.ndarray` or `tf.Tensor` </param> </returns> </return>
        List<Tensor> forward(IDatasetV2 dataset, Dictionary<string, object> kwargs = null)
        {
            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();

            foreach (var model_inputs in dataset.batch(this.args.batch_size))
            {
                var model_outputs = this.model_forward(model_inputs.Item1, mode: "test");

                foreach (var index in range(len(model_outputs)))
                {
                    if (len(model_outputs_all) < index + 1)
                    {
                        model_outputs_all.append(new List<Tensor>());
                    }
                    model_outputs_all[index].append(model_outputs[index]);
                }
            }

            List<Tensor> tensor_outputs = new List<Tensor>();
            foreach (var index in range(len(model_outputs_all)))
            {
                tensor_outputs[index] = tf.concat(model_outputs_all[index], axis: 0);
            }

            return tensor_outputs;
        }

        List<NDArray> call(Tensors model_inputs)
        {
            var test_dataset = this.load_forward_dataset(new Dictionary<string, object> { { "model_inputs", model_inputs } });
            var tensor_outputs = this.forward(test_dataset);

            List<NDArray> results = new List<NDArray>();
            foreach (var index in range(len(tensor_outputs)))
            {
                results[index] = tensor_outputs[index].numpy();
            }
            return results;
        }
    }
}