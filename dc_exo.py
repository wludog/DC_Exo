import graphviz
import pandas
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split as split


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
        self.headers = (samples.columns.values[:-1])
        self.data = samples.iloc[:, :-1].values
        # Define labels
        self.classNames = samples.iloc[:, -1].values
        ly = LabelEncoder()
        self.labels = ly.fit_transform(self.classNames)
        self.classesUnique = ly.classes_
        print(self.labels)
        self.X_train, self.X_test, self.y_train, self.y_test = split(
                                                self.data,
                                                self.labels,
                                                test_size=0.5,
                                                random_state=None,
                                                shuffle=True,
                                                stratify=self.labels)

    def classify(self):
        clf = tree.DecisionTreeClassifier(ccp_alpha=0.01)
        clf = clf.fit(self.X_train, self.y_train)
        graph_data = tree.export_graphviz(clf,
                                          out_file=None,
                                          feature_names=self.headers,
                                          class_names=self.classesUnique,
                                          filled=True, rounded=True,
                                          special_characters=True)
        graph = graphviz.Source(graph_data)
        graph.render("ExoPlanets")
        y_predict = clf.predict(self.X_test)
        print("Report:", classification_report(self.y_test, y_predict))
        print("Cohen_Kappa:", cohen_kappa_score(self.y_test, y_predict))
        print("Matthews Corr.:", matthews_corrcoef(self.y_test, y_predict))
        print(accuracy_score(self.y_test, y_predict))


if __name__ == "__main__":
    classifier = Classifier()
    classifier.import_for_classify("./dataKepAll.csv")
    classifier.classify()
