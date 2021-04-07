/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 14:33:11
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-04-06 21:44:57
 * @Description: file content
 */

using System;
using System.Collections.Generic;
using NumSharp;
using System.Linq;
using Tensorflow;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using static modules.models.helpMethods.HelpMethods;

namespace modules.models.Base
{
    abstract class Model
    {
        BaseArgs _base_args;
        Structure _training_structure;
        
        public Model(
            BaseArgs Args,
            Structure training_structure = null
        )
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

        public abstract Tensors call(Tensors inputs, bool training = false, dynamic mask = null);

        public abstract void build();

        public void load_weights(string base_dir){
            var name_path = System.IO.Path.Combine(base_dir, "names.txt");
            var names = read_txt_lines(name_path);
            foreach (var name in names){
                var npy_path = System.IO.Path.Combine(base_dir, string.Format("{0}.npy", name));
                var tf_variable = tf.Variable(np.load(npy_path));

                var var_name_list = name.Split("|||");
                var python_name = var_name_list[0];

                string var_type;
                dynamic layer_weights;
                var current_layer = getattr(this, python_name);

                if (python_name == "gcn_layers"){
                    var_type = var_name_list[2];
                    layer_weights = current_layer[string.Format("{0}", var_name_list[1])].weights;
                    
                } else {
                    var_type = var_name_list[1];
                    layer_weights = current_layer.weights;
                }
                
                int index = 0;
                foreach (var item in layer_weights){
                    if (item.Name.IndexOf(var_type) > -1){
                        break;
                    } else {
                        index ++;
                    }
                }

                if (python_name == "gcn_layers"){
                    current_layer[string.Format("{0}", var_name_list[1])].weights[index].assign(tf_variable);
                    
                } else {
                    current_layer.weights[index].assign(tf_variable);
                }

                setattr(this, python_name, current_layer);
            }
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
            var outputs = this.call(model_inputs_processed);
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
                output_processed = this.post_process_test(output_processed, model_inputs: model_inputs);
            }

            return output_processed;
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
        Model _model;
        private Tensorflow.Keras.Optimizers.OptimizerV2 opt;

        public virtual BaseArgs args
        {
            get
            {
                return this._base_args;
            }
        }

        public Model model
        {
            get
            {
                return this._model;
            }
            set
            {
                this._model = value;
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
            this.model.load_weights(model_path);
            return this.model;
        }

        ///FUNCTION_NAME: save_model
        ///<summary>
        ///        Save trained model to `save_path`.
        ///
        ///</summary>
        void save_model(string save_path)
        {
            // FIXME it should be `save_weights` rather than `save`
            // this.model.save(save_path);
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
        
        ///FUNCTION_NAME: load_dataset
        ///<summary>
        ///        Load training and val dataset.
        ///
        ///</summary>
        ///<return name="dataset_train"> train dataset, type = `tf.data.Dataset` </return>
        public virtual (Tensors dataset_train, Tensors dataset_val) load_dataset()
        {
            Tensors dataset_train = null;
            Tensors dataset_val = null;
            return (dataset_train, dataset_val);
        }

        ///FUNCTION_NAME: load_test_dataset
        ///<summary>
        ///        Load test dataset.
        ///
        ///</summary>
        public virtual Tensors load_test_dataset(Dictionary<string, object> kwargs = null)
        {
            Tensors dataset_test = null;
            return dataset_test;
        }
        
        ///FUNCTION_NAME: load_forward_dataset
        ///<summary>
        ///        Load forward dataset.
        ///
        ///</summary>
        public virtual Tensors load_forward_dataset(Dictionary<string, object> kwargs = null)
        {
            Tensors dataset_forward = null;
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
            var time_bar = dataset_test;

            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();
            Dictionary<string, List<Tensor>> loss_dict_all = new Dictionary<string, List<Tensor>>();

            var test_data = time_bar;
            {
                (var model_outputs, var loss, var loss_dict) = this.test_one_step(dataset_test, dataset_test[dataset_test.Length-1]);

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
        List<Tensor> forward(Tensors dataset, Dictionary<string, object> kwargs = null)
        {
            List<List<Tensor>> model_outputs_all = new List<List<Tensor>>();

            var model_inputs = dataset;
            {
                var model_outputs = this.model_forward(model_inputs, mode: "test");

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