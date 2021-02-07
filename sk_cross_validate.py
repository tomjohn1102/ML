from sklearn.model_selection import cross_validate
from sklearn.datasets import load_iris,load_diabetes
from sklearn import svm

class CVModelData(object):
    def __init__(self):
        """
        输入：
        isTest = True, 是否执行测试，不测试为False
        model= None, 输入模型，例如svm、lr等
        modelTarget = None, 输入模型目标：分类、回归
        X= None, 输入数据集特征
        y= None，输入数据集目标
        
        输出：
        CrossValidateScore：
            1、回归：训练时间、评估时间、方差、MAE、MSE、R2等
            2、分类：训练时间、评估时间、Precision、Recall、F1等
        """
        self.regressionScoring = ['explained_variance',
                                     'neg_mean_absolute_error',
                                     'neg_mean_squared_error',
                                     'neg_median_absolute_error',
                                     'r2']
        
        self.classificationScoring = ['precision_macro',
                                     'recall_macro',
                                     'f1_macro']
    
    # ==============================    分类测试部分    ==============================
    #测试--分类模型
    def _testClassModel(self):
        clf = svm.SVC(kernel='linear', C=1, random_state=0)
        return clf
    
    #测试--分类数据
    def _testClassData(self):
        iris = load_iris()
        X,y = iris.data,iris.target
        return X,y
    
    #测试--分类交叉验证
    def _testClassCrossValidate(self,scoring = None):
        model = self._testClassModel()
        X, y = self._testClassData()
        scores = self._crossValidate(model, X, y,scoring = scoring)
        return scores
    
    # ==============================    回归测试部分    ==============================    
    #测试--回归模型
    def _testRegModel(self):
        clf = svm.SVR(kernel='linear', C=1)
        return clf
    
    #测试--回归数据
    def _testRegData(self):
        db = load_diabetes()
        X,y = db.data,db.target
        return X,y
    
    #测试--回归交叉验证
    def _testRegCrossValidate(self,scoring = None):
        model = self._testRegModel()
        X, y = self._testRegData()
        scores = self._crossValidate(model, X, y,scoring = scoring)
        return scores
    
    # ==============================    主要方法    ==============================
    def main(self,isTest = True,model= None,modelTarget = None,X= None, y= None):
        if isTest == True:
            if modelTarget == 'Classification':
                scores = self._testClassCrossValidate(scoring = self.classificationScoring)
                return self.CrossValidateScore(scores,modelTarget = 'Classification')
            
            if modelTarget == 'Regression':
                scores = self._testRegCrossValidate(scoring = self.regressionScoring)
                return self.CrossValidateScore(scores,modelTarget = 'Regression')
            
        elif isTest == False and model != None and X != None and y != None:
            if modelTarget == 'Regression':
                return self._crossValidate(model,X, y,scoring = self.regressionScoring)
            
            elif modelTarget == 'Classification':
                return self._crossValidate(model,X, y,scoring = self.classificationScoring)
            
            else:
                print("需要明确模型目标是分类(Classification)还是回归(Regression)")
                return None
        else:
            print('crossValidate 需要传入机器学习模型model，训练数据X和目标y')
            return None
    
    
    #交叉验证方法
    def _crossValidate(self,model,X, y,scoring = None):
        scores = cross_validate(model, X, y,scoring = scoring)
        return scores
    
    
    #交叉验证结果
    def CrossValidateScore(self,scores,modelTarget):
        if scores == None or modelTarget == None:
            print("交叉验证结果需要传入scores和modelTarget")
            return None
        
        if modelTarget == 'Regression':
            fit_time = scores['fit_time']
            score_time = scores['score_time']
            explained_variance = scores['test_explained_variance']
            mean_absolute_error = scores['test_neg_mean_absolute_error']
            mean_squared_error = scores['test_neg_mean_squared_error']
            median_absolute_error = scores['test_neg_median_absolute_error']
            r2 = scores['test_r2']
            
            print("fitTime: %0.2f (+/- %0.2f)" % (fit_time.mean(), fit_time.std() * 2))
            print("scoreTime: %0.2f (+/- %0.2f)" % (score_time.mean(), score_time.std() * 2))
            print("explainedVariance: %0.2f (+/- %0.2f)" % (explained_variance.mean(), explained_variance.std() * 2))
            print("meanAbsoluteError: %0.2f (+/- %0.2f)" % (mean_absolute_error.mean(), mean_absolute_error.std() * 2))
            print("meanSquaredError: %0.2f (+/- %0.2f)" % (mean_squared_error.mean(), mean_squared_error.std() * 2))
            print("medianAbsoluteError: %0.2f (+/- %0.2f)" % (median_absolute_error.mean(), median_absolute_error.std() * 2))
            print("r2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
            
            return explained_variance,mean_absolute_error,mean_squared_error,median_absolute_error,r2
        
        if modelTarget == 'Classification':
            fit_time = scores['fit_time']
            score_time = scores['score_time']
            precision_macro = scores['test_precision_macro']
            recall_macro = scores['test_recall_macro']
            f1_macro = scores['test_f1_macro']
            
            print("fitTime: %0.2f (+/- %0.2f)" % (fit_time.mean(), fit_time.std() * 2))
            print("scoreTime: %0.2f (+/- %0.2f)" % (score_time.mean(), score_time.std() * 2))
            print("precisionMacro: %0.2f (+/- %0.2f)" % (precision_macro.mean(), precision_macro.std() * 2))
            print("recallMacro: %0.2f (+/- %0.2f)" % (recall_macro.mean(), recall_macro.std() * 2))
            print("f1Macro: %0.2f (+/- %0.2f)" % (f1_macro.mean(), f1_macro.std() * 2))
            
            return fit_time,score_time,precision_macro,recall_macro,f1_macro
            
if __name__ == "__main__":
  cv = CVModelData()
  classScores = cv.main(isTest = True,modelTarget = 'Classification')
  print("="*50)
  regScores = cv.main(isTest = True,modelTarget = 'Regression')
