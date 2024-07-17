

from survshap import SurvivalModelExplainer, ModelSurvSHAP, PredictSurvSHAP
import os
from datetime import datetime
from BaseModel import BaseModel



class Explainers(BaseModel):
    def __init__(self, survival_model,
                 pipeline_features,
                 pipline_labels,
                 model_save_path:str):

        super().__init__(model_save_path)
        if survival_model is None:
            raise ValueError("生存分析模型未训练完成")
        self.survival_model = survival_model
        self.explainer = None
        self.global_explainers = None
        self.instance_explainers = None
        self.explainer_init(pipeline_features,pipline_labels)

    def explainer_init(self,pipeline_features,pipline_labels)->None:
        self.explainer = SurvivalModelExplainer(self.survival_model,pipeline_features,pipline_labels)

    def global_explainer_init(self)->None:
        current_time = datetime.now().strftime("%Y%m%d-%H")
        model_subdir=self.get_model_subdir(model_type="Global_Explainers")
        if not os.path.exists(model_subdir):
            global_explainers=ModelSurvSHAP(random_state=42,calculation_method="treeshap")
            global_explainers.fit(self.explainer)
            self.model_save(global_explainers,"Global_Explainers")
        else:
            global_explainers=self.get_latest_model("Global_Explainers")
        self.global_explainers=global_explainers


    def instance_explainer_init(self,instance_features)->None:
        self.instance_explainers=PredictSurvSHAP()
        self.instance_explainers.fit(self.explainer,instance_features)


    def instance_shap_plot(self,instance_features)->None:
        self.instance_explainer_init(instance_features)
        self.instance_explainers.plot()


    def global_shap_plot(self)->None:
        if self.global_explainers is None:
            self.global_explainer_init()
        self.global_explainers.plot_mean_abs_shap_values()



