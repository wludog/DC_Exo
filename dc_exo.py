# comment in to plot DC tree graph import graphviz
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split as split
from sklearn.model_selection import cross_val_score, StratifiedKFold


class Classifier:
    def __init__(self):
        self.data = None
        self.labels = None
        self.headers = None
        self.classNames = None
        self.classesUnique = None

    def import_for_classify(self, filename):
        # Import data
        samples = pandas.read_csv(filename, delimiter=";", decimal=",")
        #samples = samples.drop(samples.columns[1], axis = 1)
        print(samples)
        #samples["Lum"] = samples.iloc[:, 2]**2 * samples.iloc[:, 1]**4
        samples["Ins"] = (samples.iloc[:, 3]**2 * samples.iloc[:, 2]**4)/(samples.iloc[:, 0]**2)
        #samples["SRadOverDist"] = samples.iloc[:, 3]/(samples.iloc[:, 0]**2)
        samples["InsOverSurface"] = (samples.iloc[:, 3]**2 * (samples.iloc[:, 2]**4)*(samples.iloc[:, 1]**2))/(samples.iloc[:, 0]**2)
        #samples["PradsqOverDist"] = (samples.iloc[:, 1]**2)/(samples.iloc[:, 0]**2)
        #samples["Temp*Rad/Dist"] = (samples.iloc[:, 2])*(samples.iloc[:, 3])/(samples.iloc[:, 0]**2)
        
        #samples.iloc[:, 2] = samples.iloc[:, 2]/(samples.iloc[:, 0]**2)
        samples.iloc[:, 1] = samples.iloc[:, 1]
        samples.iloc[:, 2] = samples.iloc[:, 1]
        
        #samples = samples.drop(samples.columns[0], axis = 1)
        #samples = samples.drop(samples.columns[0], axis = 1)
        #samples.insert(1, "Lum", samples.pop("Lum"))
        samples.insert(1, "Ins", samples.pop("Ins"))
        #samples.insert(1, "SRadOverDist", samples.pop("SRadOverDist"))
        samples.insert(1, "InsOverSurface", samples.pop("InsOverSurface"))
        #samples.insert(1, "PradsqOverDist", samples.pop("PradsqOverDist"))
        #samples.insert(1, "Temp*Rad/Dist", samples.pop("Temp*Rad/Dist"))
        self.samples = samples
        print(samples)
        self.headers = (samples.columns.values[:-1])
        self.data = samples.iloc[:, :-1].values
        # Define labels
        self.classNames = samples.iloc[:, -1].values
        ly = LabelEncoder()
        self.labels = ly.fit_transform(self.classNames) 
        self.classesUnique = ly.classes_
        state = np.random.randint(1,51)
        state=23 #state for water
        #state=13 #state for ammonia
        #state=23 #state for sulfuric
        #state=37 #state for carbon
        self.state = state
        print(self.state)
        self.X_train, self.X_test, self.y_train, self.y_test = split(
                                                self.data,
                                                self.labels,
                                                test_size=0.2,
                                                random_state=state,
                                                shuffle=True,
                                                stratify=self.labels)
        # smote = SMOTE(random_state=42)
        # self.X_train_sm, self.y_train_sm = smote.fit_resample(self.X_train, self.y_train)     
                                                

    def classify(self):
        #clf = tree.DecisionTreeClassifier(ccp_alpha=0.01)
        clf = RandomForestClassifier(random_state=self.state, n_estimators=1000, class_weight="balanced")
        #clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=42)
        #clf = GradientBoostingClassifier(n_estimators=1000, max_depth=20, random_state=42)
        #clf = BaggingClassifier(estimator=RandomForestClassifier(random_state=1),
        #n_estimators=1000, random_state=1)
        #clf = clf.fit(self.X_train_sm, self.y_train_sm)
        clf = clf.fit(self.X_train, self.y_train)
        importances = clf.feature_importances_
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.state)
        crossValScores = cross_val_score(clf, self.data, self.labels, cv=kfold, scoring = "recall")
        print(crossValScores)
        print(crossValScores.mean())
        print(crossValScores.std())
        print(importances)
        visual = pandas.Series(importances, index = self.headers)
        fig, ax = plt.subplots()
        visual.plot.bar(ax=ax)
        ax.set_title("Feature importances (CO2-like atm with water)")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        #plt.show()
        
        #clf = clf.fit(self.X_train, self.y_train)
        # graph_data = tree.export_graphviz(clf,
        #                                   out_file=None,
        #                                   feature_names=self.headers,
        #                                   class_names=self.classesUnique,
        #                                   filled=True, rounded=True,
        #                                   special_characters=True)
        # graph = graphviz.Source(graph_data)
        # comment this block in to plot a graph , graph.render("DC_Exo_Class")
        y_predict = clf.predict(self.X_test)
        print("Report:", classification_report(self.y_test, y_predict))
        print("Cohen_Kappa:", cohen_kappa_score(self.y_test, y_predict))
        print("Matthews Corr.:", matthews_corrcoef(self.y_test, y_predict))
        print(accuracy_score(self.y_test, y_predict))


if __name__ == "__main__":
    classifier = Classifier()
    classifier.import_for_classify("./waterexRadTemp.csv")
    #classifier.import_for_classify("./ammonRadTemp.csv")
    #classifier.import_for_classify("./sulfurRadTemp.csv")
    #classifier.import_for_classify("./carbonRadTemp.csv")
    classifier.classify()
