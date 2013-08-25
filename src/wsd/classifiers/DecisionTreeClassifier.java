package wsd.classifiers;

import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;

public class DecisionTreeClassifier {

	public static Classifier newInstance() {
		return new J48();
	}

}
