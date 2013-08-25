package wsd.utils;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import javax.xml.parsers.ParserConfigurationException;

import org.xml.sax.SAXException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import wsd.utils.DataUtil;

public abstract class TestUtil {
	
	private static final String trainingFile = "data/Chinese_train_pos.xml";
	private static final String testingFile = "data/Chinese_test_pos.xml";
	private static Map<String, Instances> trainingData;
	private static Map<String, Instances> testingData;
	
	static {
		try {
			trainingData = DataUtil.buildTrainingDataset(trainingFile);
			testingData = DataUtil.buildTestingDataset(testingFile, trainingData);
		} catch (SAXException | IOException | ParserConfigurationException e) {
			e.printStackTrace();
		}
	}
	
	public static void test(HashMap<String, Classifier> classifiers) throws Exception {
		HashMap<String, int[]> resultMap = new HashMap<String, int[]>();
		Set<String> targetWords = testingData.keySet();
		
		for (String tw : targetWords) {
			Instances testingInstances = testingData.get(tw);
			testingInstances.setClassIndex(testingInstances.numAttributes() - 1);
			Classifier classifier = classifiers.get(tw);
			classifier.buildClassifier(trainingData.get(tw));
			int numCorrect = 0;
			int[] resultArray = new int[2];
			
			for (int i = 0; i < testingInstances.numInstances(); i++) {
				Instance inst = testingInstances.instance(i);
				double clsLabel = classifier.classifyInstance(inst);
				if (clsLabel == inst.classValue()) {
					numCorrect++;
				}
			}
			resultArray[0] = numCorrect;
			resultArray[1] = testingInstances.numInstances();
			resultMap.put(tw, resultArray);
		}
		
		// macro average
		double correctPctSum = 0d;
		HashMap<String, Double> correctPctMap = new HashMap<String, Double>();
		for(String tw : targetWords) {
			int[] result = resultMap.get(tw);
			double correctPct = (double) result[0] / (double) result[1] * 100;
			correctPctSum += correctPct;
			correctPctMap.put(tw, correctPct);
		}
		List<Entry<String, Double>> sortedResult = SortUtil.sortMapDesc(correctPctMap);
		for (Entry<String, Double> e : sortedResult) {
			System.out.println(e.getKey() + " " + e.getValue());
		}
		System.out.println("\n宏平均: " + correctPctSum / (double) targetWords.size());
		
		// micro average
		int correctSum = 0;
		int total = 0;
		for(String tw : targetWords) {
			int[] result = resultMap.get(tw);
			correctSum += result[0];
			total += result[1];
		}
		System.out.println("微平均: " + (double) correctSum / (double) total * 100);
	}

	public static Map<String, Double> evaluate(Classifier classifier) throws Exception {
		Set<String> targetWords = trainingData.keySet();
		Map<String, Double> result = new HashMap<String, Double>();
		
		for (String tw : targetWords) {
			Instances data = trainingData.get(tw);
			
			Evaluation eval = new Evaluation(data);
			eval.crossValidateModel(classifier, data, 10, new Random(1));
			
			result.put(tw, eval.pctCorrect());
			System.out.println(tw + " " + eval.pctCorrect());
		}
		
		List<Entry<String, Double>> sortedResult = SortUtil.sortMapDesc(result);
		double sum = 0d;
		System.out.println("------------------------------------------------");
		
		for (Entry<String, Double> e : sortedResult) {
			String tw = e.getKey();
			double precision = e.getValue();
			int noSenses = trainingData.get(tw).attribute("sense").numValues() - 1;
			System.out.println(tw + " " + noSenses + " " + precision);
			sum += precision;
		}
		
		System.out.println("\n平均: " + sum / sortedResult.size() + "\n");
		System.out.println("------------------------------------------------");
		System.out.println("------------------------------------------------");
		System.out.println();
		return result;
	}
}
