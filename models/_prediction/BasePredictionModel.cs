/*
 * @Author: Conghao Wong
 * @Date: 2021-01-26 18:46:31
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2021-01-28 00:48:57
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
                    
                }
            }
        }

        Model load_from_checkpoint(string model_path){
            this.load_model(model_path);
            return this.model;
        }

        (List<Tensor> model_inputs, Tensor gt) prepare_model_inputs_all(List<BaseAgentManager> input_agents){
            return getInputs_onlyTraj(input_agents);
        }

        // public override (DatasetV2, DatasetV2) load_dataset(){
            
        // }
    }
}