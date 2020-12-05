#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
using namespace std::placeholders;

struct Intersection{
    Intersection(const Vec4i &parentLine1, const Vec4i &parentLine2, const Point &point) : parentLine1(parentLine1), parentLine2(parentLine2), point(point) {}

    Vec4i parentLine1;
    Vec4i parentLine2;
    Point point;
};

class EdgeDetector {
    public:
    static vector<cv::Point> detect_edges( Mat& image);
    static Mat debug_squares( Mat image );
    
    private:
    static double get_cosine_angle_between_vectors( cv::Point pt1, cv::Point pt2, cv::Point pt0 );
    static vector<vector<cv::Point> > find_squares(Mat& image);
    static float get_width(vector<cv::Point>& square);
    static float get_height(vector<cv::Point>& square);

    static vector<Vec4i> groupLines(vector<Vec4i> houghLines);
    static bool contains(vector<Vec4i> &vector, const Vec4i &line);

    static float getLineAngle(Vec4i line);

    static float getLineDistance(Vec4i line, Vec4i cmpLine);

    static float getDistance(Point a, Point b);

    static float getSlope(Vec4i vec);

    static Vec4i getShortestLineBetween(Vec4i line1, Vec4i line2);

    static Vec4i mergeLines(vector<Vec4i> lines);


    static int inputWidth;
    static int inputHeight;
    static int inputSmallerSize;

    static float anglesBetween(Vec4i line1, Vec4i line2);

    static float anglesBetween(float angle1, float angle2);

    static vector<Vec4i> removeContainedLines(vector<Vec4i> lines);

    static bool isContained(Vec4i container, Vec4i contained, float angleThreshold, float distanceThreshold);

    static vector<Vec4i> mergeOverlappingLines(vector<Vec4i> lines);

    static vector<Vec4i> getLinesWithAngle(const vector<Vec4i>& lines, float searchedAngle, float angleThreshold, vector<Vec4i> excluded);

    static vector<Vec4i> removeDistantLines(vector<Vec4i> lines, Vec4i cmpLine, int threshold);

    static Vec4i getLongestLine(vector<Vec4i> lines);

    static vector<Point> getRectangleVertecies(vector<Vec4i> lines);

    static bool isHorizontal(Vec4i line);

    static Vec4i getShortestLine(vector<Vec4i> shortest);

    static Vec4i hasMatch(vector<Vec4i> lines, Vec4i matchable, float distanceThreshold);

    static float getLength(Vec4i line);

    static Vec4i mergeMatchingLines(Vec4i line1, Vec4i line2);

    static int indexOf(vector<Vec4i> &vector, const Vec4i &line);

    static Vec4i getLongestLineBetween(Vec4i line1, Vec4i line2);

    static vector<Vec4i> findEdgeLines(vector<Vec4i> lines, int nOfLinesToFind, vector<Vec4i> oppositeDirectionLines);

    static vector<Vec4i> getParallelLines(vector<Vec4i> lines, Vec4i cmpLine, int threshold, vector<cv::Vec4i> vector);

    /*static vector<vector<Vec4i>> removeCloseParallelLines(vector<vector<Vec4i>> parallelLines, int distanceThreshold);

    static vector<vector<Vec4i>> removeLinesWithNoPairs(vector<vector<Vec4i>> parallelLines);

    static vector<vector<cv::Vec4i>> removeCloseSmallParallelLines(vector<vector<cv::Vec4i>> paralellLines, int distanceThreshold, float lengthDifferenceThreshold);*/

    static vector<Point> getPointsFromIntersections(vector<Intersection> intersections, vector<Vec4i> lines);

    /*static vector<Intersection> getIntersectionsForLine(vector<Intersection> intersections, Vec4i line, vector<cv::Point> exclude);

    static bool isLineOutOfPoints(Vec4i line, vector<Intersection> intersects);*/

    static bool containsPoint(vector<cv::Point> points, Point point);

   /* static bool isOutOfImage(Point p);

    static vector<Point> getRejectedPoints(const vector<Intersection> &intersections, const vector<Vec4i> &lines);

    static  vector<Intersection> getRemoveMiddlePoints(vector<Intersection> lineIntersections, bool isHorizontal);*/

    static bool compareIntersectionPointX(Intersection i1, Intersection i2){
        return (i1.point.x < i2.point.x);
    }

    static bool compareIntersectionPointY(Intersection i1, Intersection i2){
        return (i1.point.y < i2.point.y);
    }

    static bool comparePointsX(Point p1, Point p2){
        return (p1.x < p2.x);
    }

    static bool comparePointsY(Point p1, Point p2){
        return (p1.y < p2.y);
    }

    static bool orderPointsClockwise(Point p1, Point p2, Point centerPoint);

    static vector<Vec4i> removeLinesWithWrongConnectionAngle(vector<Vec4i> lines, Vec4i cmpLine, float angleThreshold);

    static vector<Point> mergeClosePoints(vector<Point> points);

    static Vec4i get4thLine(Vec4i lineToPair, vector<Vec4i> perpendicularLines, bool isLineHorizontal);

    static vector<vector<Point>> mergeSquares(vector<vector<Point>> polyPoints);

    static vector<vector<Point>> removeSmallRectangles(vector<vector<Point>> polyPoints);

    static vector<vector<Point>> getRectangles(vector<vector<Point>> polyPoints);

    static vector<vector<Point>> mergeIntoLargestRectangle(vector<vector<Point>> polyPoints);

    static float getPolygonArea(vector<Point> polyPoints);

    static Point getPointsCenter(vector<Point> points);

    static bool isRectangleContained(vector<Point> points, vector<Point> container);

    static bool isPointInPoly(Point point, vector<Point> polyPoints);

    static bool doIntersect(Point p1, Point q1, Point p2, Point q2);

    static int orientation(Point p, Point q, Point r);

    static bool onSegment(Point p, Point q, Point r);
};


