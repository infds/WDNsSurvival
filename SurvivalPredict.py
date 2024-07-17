import pandas as pd
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, integrated_brier_score
from survshap import SurvivalModelExplainer, ModelSurvSHAP, PredictSurvSHAP
import os
import copy
from datetime import datetime
from typing import List, Tuple
from Data.DataRecorder import PiplineRecoder,data_merge
from ShapExplainers import Explainers
from BaseModel import BaseModel




class SurvivalPredictModel(BaseModel):
    """
    该类提供随机生存森林或COX模型的生存分析实例以及SurvSHAP(t)的分析功能
    """

    def __init__(self,
                 raw_data:pd.DataFrame,
                 survival_model_type: str = "Forest",
                 model_save_path: str = "Results",
                 random_seed: int = 42,
                 feature_list: list = None,
                 **kwargs) -> None:
        super().__init__(model_save_path)
        allowed_model_types = ["Forest", "Cox"]
        if survival_model_type not in allowed_model_types:
            raise ValueError(f"Invalid survival_model_type. Expected one of {allowed_model_types}")

        self.survival_model = survival_model_type
        self.random_seed = random_seed
        self.feature_list = feature_list
        self.mask = None
        self.times = None
        self.model_params = kwargs
        self.pipeline_recoder =PiplineRecoder(raw_data=raw_data,feature_list=feature_list)
        self.explainer=None
        self.data_init()
    def data_init(self) -> None:
        pipelines=self.pipeline_recoder
        self.mask = (pipelines.get_label()["Time"] < pipelines.get_label()[pipelines.get_label()["Event"] == 1]["Time"].max()) & (
                        pipelines.get_label()["Time"] > pipelines.get_label()[pipelines.get_label()["Event"] == 1]["Time"].min())
        self.times = np.unique(np.percentile(pipelines.get_label()[self.mask]["Time"], np.linspace(0.1, 99.9, 101)))


    def model_train(self) -> None:
        if self.survival_model == "Forest":
            model = RandomSurvivalForest(**self.model_params)
        else:
            model = CoxPHSurvivalAnalysis(**self.model_params)
        save_path = self.get_model_subdir("Predict")
        if not os.path.exists(save_path):
            model.fit(self.pipeline_recoder.get_pipline_features(), self.pipeline_recoder.get_label())
            self.model_save(model, "Predict")
        else:
            model=self.get_latest_model("Predict")

        self.explainer=Explainers(survival_model=model,
                                      pipeline_features=self.pipeline_recoder.get_pipline_features(),
                                      pipline_labels=self.pipeline_recoder.get_label(),
                                      model_save_path=self.model_save_path)
        self.explainer.global_explainer_init()

    def get_survival_function(self)->Tuple[List[np.ndarray], List]:
        survival_function_dir = os.path.join(self.model_save_path, "SurvivalFunction")
        predict_results_path = os.path.join(survival_function_dir, "predict_results.csv")
        survival_functions_path = os.path.join(survival_function_dir, "survival_functions.csv")

        if os.path.exists(survival_function_dir) and os.path.isfile(predict_results_path) and os.path.isfile(
                survival_functions_path):
            print("读取文件中")
            preds = pd.read_csv(predict_results_path).values.tolist()
            survs = pd.read_csv(survival_functions_path).values.tolist()
            return preds, survs
        else:
            model = self.get_latest_model("Predict")
            survs = model.predict_survival_function(self.pipeline_recoder.get_pipline_features())
            preds = [fn(self.times) for fn in survs]
            if not os.path.exists(survival_function_dir):
                os.makedirs(survival_function_dir)
            pd.DataFrame(preds).to_csv(predict_results_path, index=False)
            pd.DataFrame(survs).to_csv(survival_functions_path, index=False)

            return preds, survs

    def model_metric(self):

        model = self.get_latest_model("Predict")
        survs = model.predict_survival_function(self.pipeline_recoder.get_pipline_features()[self.mask])
        preds = [fn(self.times) for fn in survs]
        brier_rsf = brier_score(self.pipeline_recoder.get_label(), self.pipeline_recoder.get_label()[self.mask], preds, self.times)
        integrated_bs = integrated_brier_score(self.pipeline_recoder.get_label(), self.pipeline_recoder.get_label()[self.mask], preds, self.times)
        return integrated_bs

    def find_time_below_threshold(self, threshold: float) -> List[float]:
        preds,_ = self.get_survival_function()
        times_below_threshold = []

        for pred in preds:
            time_below_threshold = None
            for time, survival_prob in zip(self.times, pred):
                if survival_prob < threshold:
                    time_below_threshold = time
                    break
            times_below_threshold.append(time_below_threshold)

        return times_below_threshold





if __name__ == "__main__":
    params = {
        "random_state": 42,
        "n_estimators": 120,
        "max_depth": 8,
        "min_samples_leaf": 4,
        "max_features": 3
    }
    raw_data=data_merge("Data\Wdns.csv","Data/repairs.csv")

    obj=SurvivalPredictModel(raw_data=raw_data, feature_list=["公称直径", "材质", "管长", "连接类型"], **params)
    obj.model_train()
    obj.explainer.global_shap_plot()
    obj.explainer.instance_shap_plot(instance_features=obj.pipeline_recoder.get_pipline_feature_by_pipline_Number("575329GX5699200"))

