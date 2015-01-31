import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DMatch;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.KeyPoint;
import org.opencv.highgui.Highgui;

public class LogoTracker 
{
	private FeatureDetector detector;
	private MatOfKeyPoint templateKeypoints;
	private DescriptorMatcher matcher;
	private DescriptorExtractor extractor;
	private Mat templateDescriptors;
	
	private ImageDisplay display;
	
	private boolean init = false;
	
	public LogoTracker()
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		display = new ImageDisplay("C:\\Users\\User\\FRCJavaProgramming\\LogoTracking\\src\\logo.jpg", 
				"C:\\Users\\User\\FRCJavaProgramming\\LogoTracking\\src\\sample1.jpg");
		
		detector = FeatureDetector.create(FeatureDetector.SIFT);
		matcher = DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
		templateDescriptors = new Mat();
		extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
		templateKeypoints = new MatOfKeyPoint();
	}
	
	public static void main(String args[])
	{
		LogoTracker tracker = new LogoTracker();
		tracker.train("C:\\Users\\User\\FRCJavaProgramming\\LogoTracking\\src\\logo.jpg");
		tracker.analyze("C:\\Users\\User\\FRCJavaProgramming\\LogoTracking\\src\\sample1.jpg");
	}
	
	public void train(String templatePath)
	{
		if(init)
		{
			System.err.println("You already trained the tracker for a target.");
			return;
		}
		Mat template = Highgui.imread(templatePath, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		detector.detect(template, templateKeypoints);
		extractor.compute(template, templateKeypoints, templateDescriptors);
		init = true;
	}
	
	public void analyze(String filename)
	{
		if(!init)
		{
			System.err.println("You must train the tracker first by calling train()");
			return;
		}
		Mat image = Highgui.imread(filename, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		MatOfKeyPoint imageKeypoints = new MatOfKeyPoint();
		detector.detect(image, imageKeypoints);
		
		Mat imageDescriptors = new Mat();
		extractor.compute(image, imageKeypoints, imageDescriptors);
		
		MatOfDMatch matches = new MatOfDMatch();
		matcher.match(templateDescriptors, imageDescriptors, matches);
				
		KeyPoint[] keypointArray = imageKeypoints.toArray();
		DMatch[] matchArray = matches.toArray();
		
		float minDistance = 100000f; // big number 
		
		for(DMatch match : matchArray)
		{
			if(match.distance < minDistance)
			{
				minDistance = match.distance;
			}
		}
		
		for(DMatch match : matchArray)
		{
			if(match.distance > 5 * minDistance)
			{
				int x1 = (int)keypointArray[match.trainIdx].pt.x;
				int y1 = (int)keypointArray[match.trainIdx].pt.y;
				int x2 = (int)keypointArray[match.queryIdx].pt.x;
				int y2 = (int)keypointArray[match.queryIdx].pt.y;
			
				display.drawMatch(x1, y1, x2, y2);
			}
		}
	}
	
}
