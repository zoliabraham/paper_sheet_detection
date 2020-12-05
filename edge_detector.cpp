#include "edge_detector.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

using namespace cv;
using namespace std;

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
double EdgeDetector::get_cosine_angle_between_vectors(cv::Point pt1, cv::Point pt2, cv::Point pt0)
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

vector<cv::Point> image_to_vector(Mat& image)
{
    int imageWidth = image.size().width;
    int imageHeight = image.size().height;

    return {
        cv::Point(0, 0),
        cv::Point(imageWidth, 0),
        cv::Point(0, imageHeight),
        cv::Point(imageWidth, imageHeight)
    };
}

vector<cv::Point> EdgeDetector::detect_edges(Mat& image)
{
    static struct sortY {
        bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.y < pt2.y);}
    } orderRectangleY;

    static struct sortX {
        bool operator() (cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x);}
    } orderRectangleX;

    vector<vector<cv::Point>> squares;
    vector<cv::Point>* biggestSquare = NULL;
    try {
        squares = find_squares(image);

        for (int i = 0; i < squares.size(); i++) {
            vector<cv::Point>* currentSquare = &squares[i];

            std::sort(currentSquare->begin(),currentSquare->end(), orderRectangleY);
            std::sort(currentSquare->begin(),currentSquare->begin()+2, orderRectangleX);
            std::sort(currentSquare->begin()+2,currentSquare->end(), orderRectangleX);

            float currentSquareWidth = get_width(*currentSquare);
            float currentSquareHeight = get_height(*currentSquare);

            if (currentSquareWidth < image.size().width / 5 || currentSquareHeight < image.size().height / 5) {
                continue;
            }

            /*if (currentSquareWidth > image.size().width * 0.99 || currentSquareHeight > image.size().height * 0.99) {
                continue;
            }*/

            if (biggestSquare == NULL) {
                biggestSquare = currentSquare;
                continue;
            }

            float biggestSquareWidth = get_width(*biggestSquare);
            float biggestSquareHeight = get_height(*biggestSquare);

            if (currentSquareWidth * currentSquareHeight >= biggestSquareWidth * biggestSquareHeight) {
                biggestSquare = currentSquare;
            }
        }
    }catch(...) {

    }


    if (biggestSquare == NULL) {
        return image_to_vector(image);
    }

    std::sort(biggestSquare->begin(),biggestSquare->end(), orderRectangleY);
    std::sort(biggestSquare->begin(),biggestSquare->begin()+2, orderRectangleX);
    std::sort(biggestSquare->begin()+2,biggestSquare->end(), orderRectangleX);

    return *biggestSquare;
}

float EdgeDetector::get_height(vector<cv::Point>& square) {
    float upperLeftToLowerRight = square[3].y - square[0].y;
    float upperRightToLowerLeft = square[1].y - square[2].y;

    return max(upperLeftToLowerRight, upperRightToLowerLeft);
}

float EdgeDetector::get_width(vector<cv::Point>& square) {
    float upperLeftToLowerRight = square[3].x - square[0].x;
    float upperRightToLowerLeft = square[1].x - square[2].x;

    return max(upperLeftToLowerRight, upperRightToLowerLeft);
}

cv::Mat EdgeDetector::debug_squares( cv::Mat image )
{
    vector<vector<cv::Point> > squares = find_squares(image);

    for (const auto & square : squares) {
        // draw rotated rect
        /*cv::RotatedRect minRect = minAreaRect(cv::Mat(square));
        cv::Point2f rect_points[4];
        minRect.points( rect_points );
        Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        for ( int j = 0; j < 4; j++ ) {
            cv::line( image, rect_points[j], rect_points[(j+1)%4], color, 10, 0 ); // blue
        }*/
        Scalar color = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        for(int i=0; i<square.size(); i++){
            cv::line( image, square[i], square[(i+1)%square.size()], color, 10, 0 );
        }
    }

    return image;
}

int EdgeDetector::inputWidth;
int EdgeDetector::inputHeight;
int EdgeDetector::inputSmallerSize;

vector<vector<cv::Point> > EdgeDetector::find_squares(Mat& image)
{
    inputWidth = image.cols;
    inputHeight = image.rows;
    inputSmallerSize = image.cols < image.rows ? image.cols : image.rows;

    vector<int> usedThresholdLevel;
    vector<vector<Point> > squares;

    Mat gray0(image.size(), CV_8U), gray;

    cvtColor(image , gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 21);      // blur will enhance edge detection


    int thresholdLevels[] = {30, 50, 70, 90};
    for(int thresholdLevel : thresholdLevels) {
        //GaussianBlur(gray, gray,Size(5,5),0);
        Canny(gray, gray0, thresholdLevel, thresholdLevel*3, 3);

        dilate(gray0, gray0, Mat(), Point(-1, -1), 1);

        vector<Vec4i> linesHough;
        vector<Vec4i> lines;
        HoughLinesP(gray0, linesHough, 1, CV_PI/180, 40, 400, 10 );

        cvtColor(gray0 , gray0, COLOR_GRAY2BGR);

        lines = groupLines(linesHough);

        Mat houghLinesImg =  Mat::zeros(gray0.rows, gray0.cols, CV_8UC3);
        for(Vec4i v: lines) {
            Point p1 = Point(v[0], v[1]);
            Point p2 = Point(v[2], v[3]);
            Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
            line(houghLinesImg, p1, p2, color, 10, LineTypes::LINE_AA);
        }
        /*for(Vec4i v: linesHough) {
            Point p1 = Point(v[0], v[1]);
            Point p2 = Point(v[2], v[3]);
            Scalar color = Scalar(100, 100, 100);
            line(houghLinesImg, p1, p2, color, 5, LineTypes::LINE_AA);
        }*/

        vector<Point> points = getRectangleVertecies(lines);
        for(Point p: points) {
            //Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
            //Scalar color = Scalar(255, 0, 0);
            Scalar color = cv::Scalar(0, 0, 255);
            circle(houghLinesImg, p, 25, color, -1);
        }

        if(points.size()==4) {
            squares.push_back(points);
        }

        Mat houghCopy = houghLinesImg.clone();
        resize(houghCopy, houghCopy, Size(), 0.2, 0.2);
        imshow(to_string(thresholdLevel), houghCopy);
    }


    if(squares.size() > 1)
        squares = mergeSquares(squares);

    return squares;
}

vector<Vec4i> EdgeDetector::groupLines(vector<Vec4i> houghLines) {
    vector<Vec4i> finalLines = houghLines;
    vector<Vec4i> processedLines;

    for (int i = 0; i < 3; ++i) {
        finalLines = mergeOverlappingLines(finalLines);
        finalLines = removeContainedLines(finalLines);
    }


    return finalLines;
}

bool EdgeDetector::contains(vector<Vec4i> &vector, const Vec4i &line) {
    return find(vector.begin(), vector.end(), line) != vector.end();
}

int EdgeDetector::indexOf(vector<Vec4i> &vector, const Vec4i &line) {
    auto it = find(vector.begin(), vector.end(), line);
    if(it != vector.end()){
        return it - vector.begin();
    }
    return -1;
}

float EdgeDetector::anglesBetween(Vec4i line1, Vec4i line2){
    float line1Angle = getLineAngle(line1);
    float line2Angle = getLineAngle(line2);
    return anglesBetween(line1Angle, line2Angle);
}

float EdgeDetector::anglesBetween(float angle1, float angle2){
    if(angle1>180)
        angle1-=180;
    if(angle2>180)
        angle2-=180;

    return abs(angle1-angle2);
}

float EdgeDetector::getLineAngle(Vec4i line){
    int p1x = line[0], p1y = line[1], p2x = line[2], p2y = line[3];

    float pi = 3.151592f;
    float x = (float)(p2x-p1x);
    float y = (float)(p2y-p1y);

    float angle = (atan(y/x) * 180.f/ pi);
    if(angle < 0){
        angle += 360;
    }

    return angle;
}

float EdgeDetector::getLineDistance(Vec4i line, Vec4i cmpLine) {
    float iterations = 20;

    if(line[0] == line[2]) //ha egy vonal tökéletesen függőleges, nem működik az alap paraméteres egyenlet
        line[0] += 1;
    if(cmpLine[0] == line[2])
        cmpLine[0] +=1;

    //y = m*x + b
    float m1 = getSlope(cmpLine);
    float x1 = cmpLine[0];
    float y1 = cmpLine[1];
    float b1 = y1 - (m1 * x1);

    float m2 = getSlope(line);
    float x2 = line[0];
    float y2 = line[1];
    float b2 = y2 - (m2 * x2);

    float start1x, end1x;
    if(cmpLine[0] < cmpLine[2]) { start1x = cmpLine[0]; end1x = cmpLine[2]; }
    else                        { start1x = cmpLine[2]; end1x = cmpLine[0]; }

    float start2x, end2x;
    if(line[0] < line[2]) { start2x = line[0]; end2x = line[2]; }
    else                  { start2x = line[2]; end2x = line[0]; }

    float resolution1 = (end1x - start1x)/iterations;
    float resolution2 = (end2x - start2x)/iterations;


    float minDist = 10000000.f;
    for (float i = start1x; i < end1x; i += resolution1) {
        for (float j = start2x; j < end2x; j += resolution2) {
            float y1_ = m1 * i + b1;
            float y2_ = m2 * j + b2;

            float distance = getDistance(Point(i, y1_), Point(j, y2_));
            if (distance <= minDist) {
                minDist = distance;
            }
        }
    }

    return minDist;
}

Vec4i EdgeDetector::getShortestLineBetween(Vec4i line1, Vec4i line2) {
    vector<Vec4i> lines;

    Point p11 = Point(line1[0],line1[1]);
    Point p12 = Point(line1[2],line1[3]);

    Point p21 = Point(line2[0],line2[1]);
    Point p22 = Point(line2[2],line2[3]);

    lines.push_back(Vec4i(p11.x,p11.y,p21.x,p21.y));
    lines.push_back(Vec4i(p11.x,p11.y,p22.x,p22.y));
    lines.push_back(Vec4i(p12.x,p12.y,p21.x,p21.y));
    lines.push_back(Vec4i(p12.x,p12.y,p22.x,p22.y));

    return getShortestLine(lines);
}

Vec4i EdgeDetector::getLongestLineBetween(Vec4i line1, Vec4i line2) {
    vector<Vec4i> lines;

    Point p11 = Point(line1[0],line1[1]);
    Point p12 = Point(line1[2],line1[3]);

    Point p21 = Point(line2[0],line2[1]);
    Point p22 = Point(line2[2],line2[3]);

    lines.push_back(Vec4i(p11.x,p11.y,p21.x,p21.y));
    lines.push_back(Vec4i(p11.x,p11.y,p22.x,p22.y));
    lines.push_back(Vec4i(p12.x,p12.y,p21.x,p21.y));
    lines.push_back(Vec4i(p12.x,p12.y,p22.x,p22.y));

    return getLongestLine(lines);
}

float EdgeDetector::getDistance(Point a, Point b){
    float x = b.x - a.x;
    float y = b.y -a.y;

    return sqrt(x*x + y*y);
}

float EdgeDetector::getSlope(Vec4i vec) {
    Point a(vec[0],vec[1]);
    Point b(vec[2],vec[3]);

    if(b.x == a.x)
        b.x++;

    float x = b.x - a.x;
    float y = b.y -a.y;
    return y/x;
}

Vec4i EdgeDetector::mergeLines(vector<Vec4i> lines) {
    Vec4i maxLine = lines[0];
    float maxLength = getLength(lines[0]);
    for(Vec4i line1 : lines){
        for(Vec4i line2 : lines){
            if(line1 != line2){
                Vec4i longestBetween = getLongestLineBetween(line1, line2);
                float length = getLength(longestBetween);
                if(length > maxLength){
                    maxLine = longestBetween;
                    maxLength = length;
                }
            }
        }
    }
    return maxLine;
}

vector<Vec4i> EdgeDetector::removeContainedLines(vector<Vec4i> lines) {
    float angleThreshold = 10.f; //TODO
    float distanceThreshold = 100.f;
    vector<Vec4i> finalLines;
    vector<Vec4i> containedLines;

    for(Vec4i line1: lines){
        if(!contains(containedLines, line1)){
            for(Vec4i line2: lines){
                if(line1!=line2 && !contains(containedLines, line2)){
                    if(isContained(line1, line2, angleThreshold, distanceThreshold)){
                        containedLines.push_back(line2);
                    }
                }
            }
        }
    }

    for(Vec4i line: lines){
        if(!contains(containedLines,line)){
            finalLines.push_back(line);
        }
    }
    return finalLines;
}

bool EdgeDetector::isContained(Vec4i container, Vec4i contained, float angleThreshold, float distanceThreshold) {
    float lengthThreshold = 0.2f;
    float containerLength = getDistance(Point(container[0], container[1]), Point(container[2],container[3]));
    float containedLength = getDistance(Point(contained[0], contained[1]), Point(contained[2], contained[3]));
    if(containedLength > containerLength)
        return false;
    if(getLineDistance(container, contained) > distanceThreshold)
        return false;
    if(anglesBetween(container, contained) > angleThreshold)
        return false;

    float distanceC1P1 = getDistance(Point(container[0], container[1]), Point(contained[0], contained[1]));
    float distanceC1P2 = getDistance(Point(container[0], container[1]), Point(contained[2], contained[3]));
    float distanceC2P1 = getDistance(Point(container[2], container[3]), Point(contained[0], contained[1]));
    float distanceC2P2 = getDistance(Point(container[2], container[3]), Point(contained[2], contained[3]));

    float smallerC1 = distanceC1P1 < distanceC1P2 ? distanceC1P1 : distanceC1P2;
    float smallerC2 = distanceC2P1 < distanceC2P2 ? distanceC2P1 : distanceC2P2;

    //return true;
    if(abs((smallerC1 + smallerC2 + containedLength) - containerLength) > containerLength*lengthThreshold)
        return false;

    return distanceC1P1 < containerLength && distanceC2P1 < containerLength &&
           distanceC1P2 < containerLength && distanceC2P2 < containerLength;

}

vector<Vec4i> EdgeDetector::mergeOverlappingLines(vector<Vec4i> lines) {
    float angleThreshold = 10.f; //TODO
    vector<Vec4i> groupedLines;
    vector<Vec4i> processedLines;

    for(Vec4i line: lines){
        if(contains(processedLines, line))
            continue;

        vector<Vec4i> similarAngleLines = getLinesWithAngle(lines, getLineAngle(line), angleThreshold*2, processedLines);
        if(similarAngleLines.size()==0){
            continue;
        }
        similarAngleLines = removeLinesWithWrongConnectionAngle(similarAngleLines, line, angleThreshold);

        Vec4i merged = mergeLines(similarAngleLines);
        groupedLines.push_back(merged);
        for(Vec4i angleLines: similarAngleLines){
            processedLines.push_back(angleLines);
        }
    }
    return groupedLines;
}

vector<Vec4i> EdgeDetector::getLinesWithAngle(const vector<Vec4i>& lines, float searchedAngle, float angleThreshold, vector<Vec4i> excluded) {
    vector<Vec4i> angleLines;
    for(Vec4i line: lines){
        if(contains(excluded,line))
            continue;

        if(anglesBetween(getLineAngle(line), searchedAngle) < angleThreshold)
            angleLines.push_back(line);
    }

    return angleLines;
}

vector<Vec4i> EdgeDetector::removeDistantLines(vector<Vec4i> lines, Vec4i cmpLine, int threshold) {
    vector<Vec4i> nearLines;

    for(Vec4i line: lines){
        if(getLineDistance(cmpLine, line) < threshold){
            nearLines.push_back(line);
        }
    }

    return nearLines;
}

Vec4i EdgeDetector::getLongestLine(vector<Vec4i> lines) {
    Vec4i max = lines[0];
    float maxLength = getDistance(Point(max[0],max[1]),Point(max[2],max[3]));

    for(Vec4i line: lines) {
        float length = getDistance(Point(line[0],line[1]),Point(line[2],line[3]));

        if(length > maxLength){
            max = line;
            maxLength = length;
        }
    }

    return max;
}

vector<Point> EdgeDetector::getRectangleVertecies(vector<Vec4i> lines) {
    vector<Point> points;
    vector<Intersection> intersections;
    vector<Vec4i> horizontal;
    vector<Vec4i> vertical;
    for(Vec4i line: lines){
        if(isHorizontal(line)){
            horizontal.push_back(line);
        } else{
            vertical.push_back(line);
        }
    }
    /*if(horizontal.size()==0){
        horizontal.push_back(Vec4i(5,5,inputWidth-5,5));
        horizontal.push_back(Vec4i(5,inputHeight-5,inputWidth-5,inputHeight-5));
    }
    if(vertical.size()==0){
        vertical.push_back(Vec4i(5,5,5,inputHeight-5));
        vertical.push_back(Vec4i(inputWidth-5,5,inputWidth-5,inputHeight-5));
    }*/


    if(lines.size()>4 && !horizontal.empty() && !vertical.empty()){
        int runCycles = vertical.size();
        for (int i = 0; i < runCycles; i++){ //vertical vonalakat összevonni 2 vonalra
            if(vertical.size()==2)
                break;

            Vec4i shortest = getShortestLine(vertical);
            float horizontalThreshold = getLength(getShortestLine(horizontal));
            Vec4i mergeMatch = hasMatch(vertical, shortest, horizontalThreshold);
            if(mergeMatch[0]!=-1){ //ha talált értéket
                Vec4i mergedLines = mergeMatchingLines(shortest, mergeMatch);
                int shortestId = indexOf(vertical, shortest);
                int matchingId = indexOf(vertical, mergeMatch);
                if(shortestId > matchingId){
                    vertical.erase(vertical.begin() + shortestId);
                    vertical.erase(vertical.begin() + matchingId);
                } else{
                    vertical.erase(vertical.begin() + matchingId);
                    vertical.erase(vertical.begin() + shortestId);
                }
                vertical.push_back(mergedLines);
            }
        }
        runCycles = horizontal.size();
        for (int i = 0; i < runCycles; i++){ //horizontal vonalakat összevonni 2 vonalra
            if(horizontal.size()==2)
                break;

            Vec4i shortest = getShortestLine(horizontal);
            float verticalThreshold = getLength(getShortestLine(vertical));
            Vec4i mergeMatch = hasMatch(vertical, shortest, verticalThreshold);
            if(mergeMatch[0]!=-1){ //ha talált értéket
                Vec4i mergedLines = mergeMatchingLines(shortest, mergeMatch);
                int shortestId = indexOf(vertical, shortest);
                int matchingId = indexOf(vertical, mergeMatch);
                if(shortestId > matchingId){
                    vertical.erase(vertical.begin() + shortestId);
                    vertical.erase(vertical.begin() + matchingId);
                } else{
                    vertical.erase(vertical.begin() + matchingId);
                    vertical.erase(vertical.begin() + shortestId);
                }
                vertical.push_back(mergedLines);
            }
        }
        if(vertical.size() > 2){
            vertical = findEdgeLines(vertical, 2, horizontal);
        }
        if(horizontal.size() >2){
            horizontal = findEdgeLines(horizontal, 2, vertical);
        }

    }
    //TODO ha <4
    if(horizontal.size() == 1){
        Vec4i borderLine = get4thLine(horizontal[0], vertical, true);
        horizontal.push_back(borderLine);
    } else if(vertical.size() == 1){
        Vec4i borderLine = get4thLine(vertical[0], horizontal, false);
        vertical.push_back(borderLine);
    }

    for(Vec4i horizontalLine: horizontal){
        for(Vec4i verticalLine: vertical){
            Point p1(horizontalLine[0], horizontalLine[1]);
            Point p2(horizontalLine[2], horizontalLine[3]);
            Point p3(verticalLine[0], verticalLine[1]);
            Point p4(verticalLine[2], verticalLine[3]);

            float a1 = p2.y - p1.y;
            float b1 = -(p2.x - p1.x);
            float c1 = p1.y * p2.x - p1.x*p2.y;

            float a2 = p4.y - p3.y;
            float b2 = -(p4.x - p3.x);
            float c2 = p3.y * p4.x - p3.x*p4.y;

            float intersectionX = ((b2 * c1) - (b1 * c2)) / ((a2 * b1) - (a1 * b2));
            float intersectionY = ((a2 * c1) - (a1 * c2)) / ((a1 * b2) - (a2 * b1));

            Intersection intersection(horizontalLine, verticalLine, Point(intersectionX, intersectionY));
            intersections.push_back(intersection);
        }
    }

    vector<Vec4i> processedLines;
    processedLines.insert(processedLines.end(), horizontal.begin(), horizontal.end());
    points = getPointsFromIntersections(intersections, processedLines);
    return points;
}

bool EdgeDetector::isHorizontal(Vec4i line) {
    //float angle = anglesBetween(0,getLineAngle(line));
    return abs(getSlope(line)) < 1.f;
}

Vec4i EdgeDetector::getShortestLine(vector<Vec4i> shortest) {
    Vec4i shortestLine = shortest[0];
    float minLength = getDistance(Point(shortestLine[0], shortestLine[1]), Point(shortestLine[2], shortestLine[3]));

    for(Vec4i line : shortest){
        float lenght = getDistance(Point(line[0], line[1]), Point(line[2], line[3]));
        if(lenght < minLength){
            shortestLine = line;
            minLength = lenght;
        }
    }

    return shortestLine;
}

Vec4i EdgeDetector::hasMatch(vector<Vec4i> lines, Vec4i matchable, float distanceThreshold) {
    int angleDiffThreshold = 30; //TODO
    float matchableAngle = getLineAngle(matchable);
    for(Vec4i line: lines){
        if(line != matchable){
            float lineAngle = getLineAngle(line);
            float angleDifference = anglesBetween(lineAngle, matchableAngle);
            Vec4i shortestBetween = getShortestLineBetween(line, matchable);
            float shortestLenght = getLength(shortestBetween);
            float shortestLengthAngle = getLineAngle(shortestBetween);
            float lineShortestAngleDiff = anglesBetween(shortestBetween, line);
            float matchableShortestAngleDiff = anglesBetween(shortestBetween, matchable);

            if(lineShortestAngleDiff < angleDiffThreshold && matchableShortestAngleDiff < angleDiffThreshold) {
                if(shortestLenght < distanceThreshold){
                    return line;
                }
            }

        }
    }
    return Vec4i(-1,-1,-1,-1);
}

float EdgeDetector::getLength(Vec4i line) {
    return getDistance(Point(line[0], line[1]), Point(line[2], line[3]));
}

Vec4i EdgeDetector::mergeMatchingLines(Vec4i line1, Vec4i line2) {
    return getLongestLineBetween(line1, line2);
}

vector<Vec4i> EdgeDetector::findEdgeLines(vector<Vec4i> lines, int nOfLinesToFind, vector<Vec4i> oppositeDirectionLines) {
    int angleThreshold = 15; //TODO
    float removeRatioThreshold =  1.f/3.f;
    vector<vector<Vec4i>> parallelLines;
    vector<Vec4i> processedLines;

    for(Vec4i line: lines){
        vector<Vec4i> parallelToLine = getParallelLines(lines, line, angleThreshold, processedLines);
        if(parallelToLine.size() >=2){
            parallelLines.push_back(parallelToLine);
            processedLines.insert(processedLines.end(), parallelToLine.begin(), parallelToLine.end());
        }
    }

    if(parallelLines.size() == 1 && parallelLines[0].size() == 2){
        return parallelLines[0];
    }

    /*parallelLines = removeCloseParallelLines(parallelLines, inputSmallerSize / 3);
    parallelLines = removeLinesWithNoPairs(parallelLines);
    parallelLines = removeCloseSmallParallelLines(parallelLines, inputSmallerSize / 3, removeRatioThreshold);*/

    if(parallelLines.size() == 1 && parallelLines[0].size() == 2){
        return parallelLines[0];
    }

    vector<Vec4i> finalLines;
    for(const vector<Vec4i>& plines : parallelLines){
        for(const Vec4i& line : plines){
            finalLines.push_back(line);
        }
    }

    return finalLines;
}

vector<Vec4i> EdgeDetector::getParallelLines(vector<Vec4i> lines, Vec4i cmpLine, int threshold, vector<cv::Vec4i> exclude) {
    vector<Vec4i> parallelLines;
    parallelLines.push_back(cmpLine);
    for(Vec4i line: lines){
        if(!contains(exclude, line) && line != cmpLine) {
            if (anglesBetween(line, cmpLine) < threshold) {
                parallelLines.push_back(line);
            }
        }
    }
    return parallelLines;
}

/*vector<vector<cv::Vec4i>> EdgeDetector::removeCloseSmallParallelLines(vector<vector<cv::Vec4i>> paralellLines, int distanceThreshold, float lengthDifferenceThreshold) {
    for(vector<Vec4i>& lines : paralellLines){
        if(lines.size() <= 2)
            continue;

        for(int i=0; i<lines.size()-2; i++){ //max-2 törölhető elem van
            if(lines.size()==2)
                break;

            Vec4i shortest = getShortestLine(lines);
            bool hasCloseParent = false;
            for(Vec4i parentLine : lines){
                if(parentLine!= shortest && getLength(getLineDistance(parentLine, shortest)) < distanceThreshold){
                    if(abs(getLength(parentLine) - getLength(shortest)) > lengthDifferenceThreshold){
                        hasCloseParent = true;
                        break;
                    }

                }
            }
            if(hasCloseParent){
                lines.erase(lines.begin() + indexOf(lines, shortest));
            } else{ //ha nincs nagy szülője a legkisebbnek, a többinek sem lesz
                break;
            }
        }
    }
    return paralellLines;
}*/

/*vector<vector<Vec4i>> EdgeDetector::removeCloseParallelLines(vector<vector<Vec4i>> parallelLines, int distanceThreshold) {
    vector<vector<Vec4i>> finalLines;

    for(vector<Vec4i> lineVector: parallelLines){
        if(lineVector.size() == 2){ //csak azokat lehet eldobni, amiből 2 van, mert lehetséges, hogy a 3-asokból az egyik a papír széle
            Vec4i shortestLine = getShortestLineBetween(lineVector[0], lineVector[1]);
            if(getLength(shortestLine) > distanceThreshold){
                finalLines.push_back(lineVector);
            }
        }
        else {
            finalLines.push_back(lineVector);
        }
    }

    return finalLines;
}*/

/*vector<vector<Vec4i>> EdgeDetector::removeLinesWithNoPairs(vector<vector<Vec4i>> parallelLines) {
    vector<vector<Vec4i>> removed;

    for(vector<Vec4i> lines : parallelLines){
        if(lines.size() != 1){
            removed.push_back(lines);
        }
    }
    return removed;
}*/

vector<Point> EdgeDetector::getPointsFromIntersections(vector<Intersection> intersections, vector<Vec4i> lines) {
    vector<Point> finalPoints;
    vector<Intersection> finalIntersections;
    vector<Point> rejectedPoints;// = getRejectedPoints(intersections, lines);

    for(Intersection intersection: intersections){
        if(!containsPoint(rejectedPoints, intersection.point)){
            finalIntersections.push_back(intersection);
        }
    }

    for(Intersection intersection: finalIntersections){
        if(!containsPoint(rejectedPoints, intersection.point)){
            finalPoints.push_back(intersection.point);
        }
    }

    finalPoints = mergeClosePoints(finalPoints);
    return finalPoints;
}

/*vector<Point> EdgeDetector::getRejectedPoints(const vector<Intersection> &intersections, const vector<Vec4i> &lines) {
    vector<Point> rejectedPoints;
    vector<Intersection> rejectedIntersections;
    for(Vec4i line: lines){
        vector<Intersection> lineIntersects = getIntersectionsForLine(intersections, line, rejectedPoints);
        if(lineIntersects.size()==0){
            continue;
        }
        if(isLineOutOfPoints(line, lineIntersects)){
            for(Intersection intersection: lineIntersects){
                rejectedPoints.push_back(intersection.point);
                rejectedIntersections.push_back(intersection);
            }
            continue;
        }

        if(lineIntersects.size() > 2){
            vector<Intersection> middleRemoved = getRemoveMiddlePoints(lineIntersects, isHorizontal(line));
            for(Intersection intersection: middleRemoved){
                rejectedIntersections.push_back(intersection);
                rejectedPoints.push_back(intersection.point);
            }
        }

        for(Intersection intersection : lineIntersects){
            if(isOutOfImage(intersection.point)) {
                rejectedPoints.push_back(intersection.point);
                rejectedIntersections.push_back(intersection);
            }
        }
    }

    bool hasRemoved = true;
    while(hasRemoved){
        hasRemoved = false;
        vector<Intersection> dependentRejectedIntersections;
        vector<Point> dependentRejectedPoints;
        for(Intersection intersection : rejectedIntersections) {
            vector<Intersection> line1Intersections = getIntersectionsForLine(intersections, intersection.parentLine1, rejectedPoints);
            if(line1Intersections.size() == 1){
                dependentRejectedIntersections.push_back(line1Intersections[0]);
                dependentRejectedPoints.push_back(line1Intersections[0].point);
                hasRemoved = true;
            }
            vector<Intersection> line2Intersections = getIntersectionsForLine(intersections, intersection.parentLine2, rejectedPoints);
            if(line2Intersections.size() == 1){
                dependentRejectedIntersections.push_back(line2Intersections[0]);
                dependentRejectedPoints.push_back(line2Intersections[0].point);
                hasRemoved = true;
            }
        }
        if(hasRemoved){
            rejectedIntersections.insert(rejectedIntersections.end(), dependentRejectedIntersections.begin(), dependentRejectedIntersections.end());
            rejectedPoints.insert(rejectedPoints.end(), dependentRejectedPoints.begin(), dependentRejectedPoints.end());
        }
    }

    return rejectedPoints;
}*/

/*vector<Intersection> EdgeDetector::getIntersectionsForLine(vector<Intersection> intersections, Vec4i line, vector<cv::Point> exclude) {
    vector<Intersection> lineIntersections;

    for(Intersection intersection : intersections){
        if(!containsPoint(exclude, intersection.point)) {
            if (intersection.parentLine1 == line || intersection.parentLine2 == line) {
                lineIntersections.push_back(intersection);
            }
        }
    }
    return lineIntersections;
}*/

bool EdgeDetector::containsPoint(vector<Point> points, Point point) {
    for(Point p : points){
        if(p.x == point.x && p.y == point.y){
            return true;
        }
    }
    return false;
}

/*bool EdgeDetector::isLineOutOfPoints(Vec4i line, vector<Intersection> intersects) {
    bool isLineHorizontal = isHorizontal(line);
    float lineP1 = isLineHorizontal? line[0] : line[1];
    float lineP2 = isLineHorizontal? line[2] : line[3];

    float prevPosition = isLineHorizontal? intersects[0].point.x : intersects[0].point.y;
    int direction;
    if(prevPosition > lineP1 && prevPosition > lineP2){
        direction = 1;
    }
    else if(prevPosition < lineP1 && prevPosition < lineP2){
        direction = -1;
    }
    else{
        return false;
    }

    for(auto & intersect : intersects){
        float position =  isLineHorizontal? intersect.point.x : intersect.point.y;
        if(direction ==1){
            if(position < lineP1 && position < lineP2){
                return false;
            }
        }
        else {
            if (position > lineP1 && position > lineP2) {
                return false;
            }
        }
    }
    return true;
}*/

/*bool EdgeDetector::isOutOfImage(Point p) {
    if(p.x < 0 || p.x > inputWidth){
        return true;
    }
    if(p.y < 0 || p.x > inputHeight){
        return true;
    }
    return false;
}

vector<Intersection> EdgeDetector::getRemoveMiddlePoints(vector<Intersection> lineIntersections,bool isLineHorizontal) {
    vector<Intersection> removedIntersections;

    if(isLineHorizontal)
        sort(lineIntersections.begin(), lineIntersections.end(), compareIntersectionPointX);
    else
        sort(lineIntersections.begin(), lineIntersections.end(), compareIntersectionPointY);

    removedIntersections.insert(removedIntersections.end(), lineIntersections.begin()+1, lineIntersections.end()-1);

    return removedIntersections;
}*/

vector<Vec4i> EdgeDetector::removeLinesWithWrongConnectionAngle(vector<Vec4i> lines, Vec4i cmpLine, float angleThreshold) {
    vector<Vec4i> finalLines;
    finalLines.push_back(cmpLine);

    for(Vec4i line : lines){
        if(line == cmpLine)
            continue;

        Vec4i shortestBetween = getShortestLineBetween(line, cmpLine);
        Vec4i longestBetween = getLongestLineBetween(line, cmpLine);

        float cmpAngleShortestDiff = anglesBetween(cmpLine, shortestBetween);
        float cmpAngleLongestDiff = anglesBetween(cmpLine, longestBetween);

        float lineAngleShortestDiff = anglesBetween(line, shortestBetween);
        float lineAngleLongestDiff = anglesBetween(line, longestBetween);

        if(cmpAngleShortestDiff < angleThreshold && cmpAngleLongestDiff < angleThreshold &&
           lineAngleShortestDiff < angleThreshold && lineAngleLongestDiff < angleThreshold){
            finalLines.push_back(line);
        }
    }

    return finalLines;
}

vector<Point> EdgeDetector::mergeClosePoints(vector<Point> points) {
    vector<Point> finalPoints;
    vector<Point> processedPoints;
    float mergeDistancePercent = 0.05f;
    for(Point p1 : points){
        if(containsPoint(processedPoints, p1))
            continue;

        vector<Point> mergePoints;
        mergePoints.push_back(p1);
        processedPoints.push_back(p1);
        for(Point p2 : points){
            if(p1!=p2 && !containsPoint(processedPoints, p2)){
                float distance = getDistance(p1,p2);
                if(distance < inputSmallerSize*mergeDistancePercent){
                    mergePoints.push_back(p2);
                    processedPoints.push_back(p2);
                }
            }
        }

        float sumX=0;
        float sumY=0;
        for(Point pMerge : mergePoints){
            sumX += pMerge.x;
            sumY += pMerge.y;
        }

        finalPoints.push_back(Point(sumX / mergePoints.size(), sumY / mergePoints.size()));
    }
    return finalPoints;
}

Vec4i EdgeDetector::get4thLine(Vec4i lineToPair, vector<Vec4i> perpendicularLines, bool isLineHorizontal) {
    vector<Point> perpPoints;
    for(Vec4i perpLine : perpendicularLines){
        perpPoints.push_back(Point(perpLine[0], perpLine[1]));
        perpPoints.push_back(Point(perpLine[2], perpLine[3]));
    }

    if(perpPoints.size() ==0)
        return Vec4i();

    sort(perpPoints.begin(), perpPoints.end(), isLineHorizontal ? comparePointsY : comparePointsX);
    bool isStart;
    if(isLineHorizontal){
        isStart = perpPoints[perpPoints.size() / 2].y <= lineToPair[0];
    }else{
        isStart = perpPoints[perpPoints.size() / 2].x <= lineToPair[1];
    }

    Point p0 = isStart? perpPoints[0] : perpPoints[perpPoints.size()-1];
    float slope = getSlope(lineToPair);

    float p1x = p0.x - inputWidth;
    float p1y = p0.y - (float)inputWidth*slope;

    float p2x = p0.x + inputWidth;
    float p2y = p0.y + (float)inputWidth*slope;

    return Vec4i(p1x, p1y, p2x, p2y);
}

vector<vector<Point>> EdgeDetector::mergeSquares(vector<vector<Point>> polyPoints) {
    polyPoints = getRectangles(polyPoints);
    if(polyPoints.empty())
        return vector<vector<Point>>();

    polyPoints = removeSmallRectangles(polyPoints);
    if(polyPoints.size() <= 1 )
        return polyPoints;
    polyPoints = mergeIntoLargestRectangle(polyPoints);

    return polyPoints;
}

vector<vector<Point>> EdgeDetector::removeSmallRectangles(vector<vector<Point>> polyPoints) {
    vector<vector<Point>> finalRectangles;
    for(vector<Point> poly : polyPoints ){
        vector<Vec4i> lines;
        for(Point p1 : poly){
            for(Point p2 : poly){
                if(p1 != p2){
                    lines.push_back(Vec4i(p1.x, p1.y, p2.x, p2.y));
                }
            }
        }
        Vec4i shortest = getShortestLine(lines);

        if(getLength(shortest) > inputSmallerSize*0.3){
            finalRectangles.push_back(poly);
        }
    }

    return finalRectangles;
}

vector<vector<Point>> EdgeDetector::getRectangles(vector<vector<Point>> polyPoints) {
    vector<vector<Point>> rectangles;
    for(vector<Point> poly : polyPoints ){
        if(poly.size() ==4){
            rectangles.push_back(poly);
        }
    }
    return rectangles;
}

vector<vector<Point>> EdgeDetector::mergeIntoLargestRectangle(vector<vector<Point>> polyPoints) {
    float maxArea = getPolygonArea(polyPoints[0]);
    vector<Point> maxAreaPoly = polyPoints[0];
    for(vector<Point> points : polyPoints){
        float area = getPolygonArea(points);
        if(area > maxArea){
            maxArea = area;
            maxAreaPoly = polyPoints[0];
        }
    }

    vector<vector<Point>> mergedPolys;
    mergedPolys.push_back(maxAreaPoly);

    /*for(vector<Point> points : polyPoints){
        if(points != maxAreaPoly){
            if(!isRectangleContained(points, maxAreaPoly)){
                mergedPolys.push_back(points);
            }
        }
    }*/

    return mergedPolys;
}

float EdgeDetector::getPolygonArea(vector<Point>polyPoints) {
    sort(polyPoints.begin(), polyPoints.end(), bind(orderPointsClockwise, _1, _2, getPointsCenter(polyPoints)));
    float area = 0;

    for (int i = 0; i < polyPoints.size(); ++i) {
        area += (float)(polyPoints[i].x * polyPoints[(i+1)%polyPoints.size()].y - polyPoints[i].y * polyPoints[(i+1)%polyPoints.size()].x);
    }


    return area/2.f;
}

Point EdgeDetector::getPointsCenter(vector<Point> points) {
    Point center(0,0);

    for(Point p: points){
        center.x += p.x;
        center.y += p.y;

    }

    center.x /= points.size();
    center.y /= points.size();
    return center;
}

bool EdgeDetector::isRectangleContained(vector<Point> points, vector<Point> container) {
    for (Point p : points) {
        if(!isPointInPoly(p, container)){
            return false;
        }
    }
    return true;
}

bool EdgeDetector::isPointInPoly(Point point, vector<Point> polyPoints) {
    Point outsidePoint(100000, point.y);
    int intersectCount = 0;
    for (int i = 0; i < polyPoints.size(); ++i) {
        int next = (i+1)%polyPoints.size();
        if(doIntersect(polyPoints[i], polyPoints[next], point, outsidePoint)){
            if (orientation(polyPoints[i], point, polyPoints[next]) == 0)
                return onSegment(polyPoints[i], point, polyPoints[next]);
            intersectCount++;
        }
    }

    return intersectCount%2 == 1;
}

bool EdgeDetector::doIntersect(Point p1, Point q1, Point p2, Point q2) {
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;

    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;

    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;

    // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;

    return false; // Doesn't fall in any of the above cases
}

int EdgeDetector::orientation(Point p, Point q, Point r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}

bool EdgeDetector::onSegment(Point p, Point q, Point r)
{
    return q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
           q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y);
}



bool EdgeDetector::orderPointsClockwise(Point p1, Point p2, Point centerPoint) {
    float angle1 = atan2(p1.y - centerPoint.y, p1.x - centerPoint.x);
    float angle2 = atan2(p2.y - centerPoint.y, p2.x - centerPoint.x);
    if(angle1 == angle2){
        return sqrt(pow(p1.x - centerPoint.x, 2) + pow(p1.y - centerPoint.y, 2)) < sqrt(pow(p2.x - centerPoint.x, 2) + pow(p2.y - centerPoint.y, 2));
    }
    else{
        return angle1 < angle2;
    }
}


