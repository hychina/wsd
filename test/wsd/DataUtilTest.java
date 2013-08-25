package wsd;

import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;

import org.junit.Test;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import wsd.utils.DataUtil;

public class DataUtilTest {

	@Test
	public void testBuildDataset() throws Exception {
		
		Map<String, Instances> trainingDatasetByTargetWord = DataUtil.buildTrainingDataset("data/Chinese_train_pos.xml");
		Map<String, Instances> testingDatasetByTargetWord = DataUtil.buildTestingDataset("data/Chinese_test_pos.xml", trainingDatasetByTargetWord);
		
		Set<String> targetWords = testingDatasetByTargetWord.keySet();
		for (String tw : targetWords) {
			Instances instances = testingDatasetByTargetWord.get(tw);
			DataSink.write("data/" + tw + ".arff", instances);
		}
	}
	
	public static void main(String[] args) {
		int count = 0;
		int lastTime = (int) (System.currentTimeMillis() / 1000);
		while (true) {
			Random random = new Random();
			double d = random.nextDouble();
			d = d * (d + 1d);
			count++;
			int currentTime = (int) (System.currentTimeMillis() / 1000);
			if (currentTime != lastTime) {
				System.out.println(count);
				System.out.println(d);
				count = 0;
			}
			lastTime = currentTime;
		}
	}
}
