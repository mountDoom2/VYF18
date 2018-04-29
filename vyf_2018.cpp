/*
 * VYF - Semetral project
 * Color Balance and Fusion for Underwater Image Enhancement
 * Based on https://ieeexplore.ieee.org/document/8058463/
 *
 * Author: Milan Skala, xskala09@stud.fit.vutbr.cz
 *
 * Complilation command: $ g++ vyf_2018.cpp -o vyf_2018 `pkg-config --cflags --libs opencv`
 *
 */
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <opencv2/photo.hpp>
#include <iostream>
using namespace cv;
using namespace std;

class Pyramid{
	public:
		static vector<Mat> GaussianPyramid(Mat img, int level, bool apply_mask=true) {
			vector<Mat> gauss_pyr;
			Mat mask, tmp, tmp1;

			if (apply_mask){
				mask = Pyramid::mask(img);
				filter2D(img, tmp, -1, mask);
			}

			gauss_pyr.push_back(tmp.clone());
			tmp = img.clone();

			for (int i = 1; i < level; i++) {
				// resize image to half size
				resize(tmp, tmp, Size(), 0.5, 0.5, INTER_LINEAR);
				// blur image
				if (apply_mask){
					filter2D(tmp, tmp1, -1, mask);
				}
				gauss_pyr.push_back(tmp1.clone());
			}
			return gauss_pyr;
		}

	static vector<Mat> LaplacianPyramid(Mat img, int level) {
		vector<Mat> lapl_pyr;

		lapl_pyr.push_back(img.clone());

		Mat tmp = img.clone();
		Mat tmp_level;
		// Get image Gaussians
		for (int i = 1; i < level; i++) {
			// resize image
			resize(tmp, tmp, Size(), 0.5, 0.5, INTER_LINEAR);
			lapl_pyr.push_back(tmp.clone());
		}

		// Laplacian pyramid can be calculated as difference of Gaussians
		for (int i = 0; i < level - 1; i++) {

			resize(lapl_pyr.at(i + 1), tmp_level, lapl_pyr.at(i).size(), 0, 0, INTER_LINEAR);
			subtract(lapl_pyr.at(i), tmp_level, lapl_pyr.at(i));
		}
		return lapl_pyr;
	}
	static Mat PyramidReconstruct(vector<Mat> pyramid) {
		int level = pyramid.size();
		Mat tmp_level;

		for (int i = level - 1; i > 0; i--) {
			resize(pyramid[i], tmp_level, pyramid.at(i - 1).size(), 0, 0, INTER_LINEAR);
			add(pyramid[i - 1], tmp_level, pyramid.at(i - 1));
		}
		return pyramid.at(0);
	}
	private:
		static Mat mask(Mat img) {
			double h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
			Mat mask = Mat(5, 5, img.type());
			for (int i = 0; i < 5; i++) {
				for (int j = 0; j <  5; j++) {
					mask.at<float>(i,j) = h[i]*h[j];
				}
			}
			return mask;
		}
};

Mat simple_whitebalance(Mat img, int percent) {
	Mat array, out;
	vector<Mat> channels, results;

	if (percent <= 0)
		percent = 5;

	img.convertTo(img, CV_32F);
	int chnls = img.channels();

    double half_percent = percent / 200.0f;
    split(img, channels);

    for(int i=0;i<chnls;i++) {
        // Find the low and high precentile values

        channels.at(i).reshape(1,1).copyTo(array);

        cv::sort(array, array, CV_SORT_ASCENDING);

        int lowval = array.at<float>(cvFloor(((float)array.cols) * half_percent));
        int highval = array.at<float>(cvCeil(((float)array.cols) * (1.0 - half_percent)));

        // Saturate pixels
        channels[i].setTo(lowval,channels[i] < lowval);
        channels[i].setTo(highval,channels[i] > highval);

        normalize(channels[i],channels[i],0,255,NORM_MINMAX);

    }
    merge(channels,out);
    return out;
}

/*
 * Equalize histogram
 * @param img: Input image
 * @param L: img luminance
 */
vector<Mat> equalize_hist(Mat img, Mat L) {
	vector<Mat> lab, result, tmp;
	Mat L2, LabIm2, img2;
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(2.0);
	clahe->apply(L, L2);

	split(img, lab);
	tmp.push_back(L2); tmp.push_back(lab.at(1)); tmp.push_back(lab.at(2));
	merge(tmp, LabIm2);
	cvtColor(LabIm2, img2, COLOR_Lab2BGR);
	result.push_back(img2);
	result.push_back(L2);
    return result;
}

/*
 * Calculate Laplacian contrast
 * Laplacian contrast is global, because is does not know
 * difference between flat and ramp areas.
 */
Mat weight_laplacian_contrast(Mat img) {
	Mat laplacian;
	Laplacian(img, laplacian, img.depth());
	convertScaleAbs(laplacian, laplacian);
	return laplacian;
}

/*
 * Calculate local contrast to preserve edges.
 * It is standard deviation between pixel's luminance value
 * and the local average of its neighborhood.
 * Uses PI/2.75 to cut off high frequencies.
 *
 */
Mat weight_local_contrast(Mat img) {
	double h[5] = { 1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0 };
	img.convertTo(img, CV_64F);
	Mat mask = Mat(5, 5, img.type());
	for (int i = 0; i < 5; i++) {
		for (int j = 0; j < 5; j++){
			mask.at<double>(i,j) = h[i]*h[j];
		}
	}
	Mat localContrast;
	localContrast.convertTo(localContrast, CV_64F);

	filter2D(img, localContrast, img.depth(), mask);
	for (int i = 0; i < localContrast.rows; i++) {
		for (int j = 0; j < localContrast.cols; j++) {
			//cout << i << " " << j << endl;
			Vec3b color = localContrast.at<Vec3b>(i,j);

			if (color[0] > M_PI / 2.75)
				localContrast.at<double>(i,j) = M_PI / 2.75;
		}
	}
	subtract(img, localContrast, localContrast);
	localContrast = localContrast.mul(localContrast);

	return localContrast;

}

/*
 * Calculate saliency, so less visible objects
 * are not lost completely.
 *
 * https://infoscience.epfl.ch/record/135217/files/1708.pdf
 * http://ivrlwww.epfl.ch/supplementary_material/RK_CVPR09/
 */
Mat weight_saliency(Mat img) {
	// blur image with a 3x3 or 5x5 Gaussian filter
	Mat gfbgr;
	GaussianBlur(img, gfbgr, Size(3, 3), 3);
	// Perform sRGB to CIE Lab color space conversion
	Mat LabIm;
	cvtColor(gfbgr, LabIm, COLOR_BGR2Lab);
	// Compute Lab average values (note that in the paper this average is found from the un-blurred original image, but the results are quite similar)
	vector<Mat> lab;
	split(LabIm, lab);
	Mat l = lab.at(0);
	l.convertTo(l, CV_32F);
	Mat a = lab.at(1);
	a.convertTo(a, CV_32F);
	Mat b = lab.at(2);
	b.convertTo(b, CV_32F);
	double lm = mean(l).val[0];
	double am = mean(a).val[0];
	double bm = mean(b).val[0];
	// Finally compute the saliency map
	Mat sm = Mat(l.rows, l.cols, l.type(), cvScalar(0.));
	subtract(l, Scalar(lm), l);
	subtract(a, Scalar(am), a);
	subtract(b, Scalar(bm), b);
	add(sm, l.mul(l), sm);
	add(sm, a.mul(a), sm);
	add(sm, b.mul(b), sm);
	return sm;
}

/*
 * Calculate pixel saturation
 */
Mat weight_saturation(Mat img){

	int rows = img.rows;
	int cols = img.cols;
	Mat saturation = Mat(rows, cols, img.type(), cvScalar(0.));
	saturation.convertTo(saturation, CV_64F);

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {

			Vec3b color = img.at<Vec3b>(Point(i,j));

			double luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
			double pow_r = pow(color[0] - luminance, 2);
			double pow_g = pow(color[1] - luminance, 2);
			double pow_b = pow(color[2] - luminance, 2);

			double value = sqrt(1./3*(pow_r + pow_g + pow_b));
			//double value = exp(-1.0 * pow(color[0] - average, 2.0) / (2 * pow(sigma, 2.0)));

			saturation.at<double>(Point(i,j)) = value;

		}
	}
	return saturation;
}

/*
 * Calculate pixel expodeness
 * Well exposed pixel have their values
 * close to the value of 0.5.
 */
Mat weight_exposedness(Mat img) {
	double sigma = 0.25;
	double average = 0.5;
	int rows = img.rows;
	int cols = img.cols;
	Mat exposedness = Mat(rows, cols, img.type(), cvScalar(0.));
	exposedness.convertTo(exposedness, CV_64F);

	for (int i = 0; i < cols; i++) {
		for (int j = 0; j < rows; j++) {

			Vec3b color = img.at<Vec3b>(Point(i,j));


			double value = exp(-1.0 * pow(color[0] - average, 2.0) / (2 * pow(sigma, 2.0)));

			exposedness.at<double>(Point(i,j)) = value;

		}
	}
	return exposedness;
}


/*
 * Compute weight matrices for color and contrast
 * enhancement. Uses Laplacian contrast nad Local contrast.
 * Saliency, saturation and expodeness for color enhancement.
 *
 * @param img: Input image
 * @param L: img luminance
 *
 * @todo: Debug contrast weights
 */
Mat calWeight(Mat img, Mat L) {
	// Normalize luminance so it contains values from <0,1>
	divide(L, Scalar(255.0), L);


	L.convertTo(L, CV_32F);
	//// calculate laplacian contrast weight
	Mat lapl_contrast = weight_laplacian_contrast(L);
	lapl_contrast.convertTo(lapl_contrast, L.type());
	//// calculate Local contrast weight
	Mat local_contrast = weight_local_contrast(L);
	local_contrast.convertTo(local_contrast, L.type());
	//// calculate the saliency weight
	Mat saliency = weight_saliency(img);
	saliency.convertTo(saliency, L.type());
	Mat expodeness = weight_exposedness(L);
	expodeness.convertTo(expodeness, L.type());
	//// calculate the saturation weight
	Mat saturation = weight_saturation(L);
	saturation.convertTo(saturation, L.type());

	// Just sum weights
	Mat weight = lapl_contrast.clone();
	add(weight, local_contrast, weight);
	add(weight, saliency, weight);
	add(weight, saturation, weight);
	add(weight, expodeness, weight);

	return weight;
}

/*
 * @param w1: Weight of white balanced image
 * @param img1: White balanced image
 * @param w2: Weight of image with reduced noise and CLAHE applied
 * @param img2: Image with reduced noise and CLAHE applied
 * @param level: Level of the Gaussian nd Laplacian pyramids
 */
Mat fuse_images(Mat w1, Mat img1, Mat w2, Mat img2, int level){
	Mat fusion;
	vector<Mat> weight1 = Pyramid::GaussianPyramid(w1, level);
	vector<Mat> weight2 = Pyramid::GaussianPyramid(w2, level);

	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);

	vector<Mat> bgr;
	split(img1, bgr);

	vector<Mat> bchnl1 = Pyramid::LaplacianPyramid(bgr.at(0), level);
	vector<Mat> gchnl1 = Pyramid::LaplacianPyramid(bgr.at(1), level);
	vector<Mat> rchnl1 = Pyramid::LaplacianPyramid(bgr.at(2), level);

	bgr.clear();
	split(img2, bgr);
	vector<Mat> bchnl2 = Pyramid::LaplacianPyramid(bgr.at(0), level);
	vector<Mat> gchnl2 = Pyramid::LaplacianPyramid(bgr.at(1), level);
	vector<Mat> rchnl2 = Pyramid::LaplacianPyramid(bgr.at(2), level);

	vector<Mat> bchnl;
	vector<Mat> gchnl;
	vector<Mat> rchnl;

	// Weights are applied on the each level separately
	for (int i = 0; i < level; i++) {
		Mat cn;

		add(bchnl1[i].mul(weight1[i]), bchnl2[i].mul(weight2[i]), cn);
		bchnl.push_back(cn.clone());

		add(gchnl1[i].mul(weight1[i]), gchnl2[i].mul(weight2[i]), cn);
		gchnl.push_back(cn.clone());
		add(rchnl1[i].mul(weight1[i]), rchnl2[i].mul(weight2[i]), cn);
		rchnl.push_back(cn.clone());

	}

	Mat bChannel = Pyramid::PyramidReconstruct(bchnl);
	Mat gChannel = Pyramid::PyramidReconstruct(gchnl);
	Mat rChannel = Pyramid::PyramidReconstruct(rchnl);

	vector<Mat> tmp;
	tmp.push_back(bChannel); tmp.push_back(gChannel); tmp.push_back(rChannel);

	merge(tmp, fusion);
	return fusion;
}

int main(int argc, char **argv) {
	if (argc != 3){
		fprintf(stderr, "Usage: ./program <input> <output>\n");
		return -1;
	}

	int pyramid_level = 5;
	Mat input_image, Lab_image;
	Mat white_balanced, white_balanced_luminance;
	Mat equalized_luminance, equalized;
	Mat w1, w2, sum_w, final;

	// Open input image
    input_image = imread(argv[1]);

    // Perform white balancing - does the most
    // notable change, but also adds noise.
    // second parameter determines
    // white balance threshold (zero = default threshold)
    white_balanced = simple_whitebalance(input_image, 0);
    white_balanced.convertTo(white_balanced, CV_8UC1);
    // Denoise image
    // http://www.ipol.im/pub/art/2011/bcm_nlm/
    cv::fastNlMeansDenoisingColored(white_balanced, white_balanced,
    								4, 10, 7, 21);

    // Convert to L*a*b*, so we can extract luminance and perform
    // histogram equalization (CLAHE)
    cvtColor(white_balanced, Lab_image, COLOR_BGR2Lab);
    extractChannel(Lab_image, white_balanced_luminance, 0);

    vector<Mat> clahe_result = equalize_hist(Lab_image, white_balanced_luminance);
    equalized = clahe_result.at(0);
    equalized_luminance = clahe_result.at(1);

    // Calculate weight matrices for latter contrast and color enhancement
    // Each weight consists of sevelar subweights which are added
    // and normalized
	w1 = calWeight(white_balanced, white_balanced_luminance);
	w2 = calWeight(equalized, equalized_luminance);

	add(w1, w2, sum_w);
	// Normalize
	divide(w1, sum_w, w1);
	divide(w2, sum_w, w2);

	// Finally, fuse images
	final = fuse_images(w1, white_balanced, w2, equalized, pyramid_level);

    imwrite(argv[2], final);

    return 0;
}





