#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc.hpp>

#include <bitplanes/core/debug.h>
#include <bitplanes/core/homography.h>
#include <bitplanes/core/bitplanes_tracker_pyramid.h>
#include <bitplanes/core/viz.h>

#include <bitplanes/utils/config_file.h>
#include <bitplanes/utils/error.h>
#include <bitplanes/utils/timer.h>

#include <boost/circular_buffer.hpp>
#include <boost/call_traits.hpp>

#include <thread>
#include <mutex>
#include <chrono>
#include <condition_variable>

#if 0

//
// from boost examples
// http://www.boost.org/doc/libs/1_60_0/doc/html/circular_buffer/examples.html
//
template <class T>
class BoundedBuffer
{
 public:
  typedef boost::circular_buffer<T> Container_t;
  typedef typename Container_t::size_type  size_type;
  typedef typename Container_t::value_type value_type;
  // call_traits is useless in this department
  typedef typename boost::call_traits<value_type>::param_type param_type;
  //typedef value_type&& param_type;

 public:
  /**
   * Initializers a buffers with at most 'capacity' elements
   */
  explicit BoundedBuffer(size_type capacity);

  /**
   * pushes an element into the buffer
   * If the buffer is full, the function waits until a slot is available
   */
  void push(param_type item);

  /**
   * set 'item' to the first element that was pushed into the buffer.
   *
   * The function waits for the specified milliseconds for data if the buffer
   * was empty
   *
   * \return true if we popped something, false otherwise (timer has gone off)
   */
  bool pop(value_type* item, int wait_time_ms = 1);

  /**
   * \return true if the buffer is full (if possible)
   */
  bool full();

  /**
   * \return the size of the buffer (number of elements) if we are able to get a
   * lock. If not, we return -1
   */
  int size();

 private:
  BoundedBuffer(const BoundedBuffer&) = delete;
  BoundedBuffer& operator=(const BoundedBuffer&) = delete;

 private:
  size_type _unread;
  Container_t _container;
  std::mutex _mutex;
  std::condition_variable _cond_not_empty;
  std::condition_variable _cond_not_full;
}; // BoundedBuffer


cv::Mat gImage; //< live image
cv::Rect gROI;  //< template ROI
cv::Point gOrigin; //< origin
volatile bool gStartSelection = false;
volatile bool gHasTemplate = false;

BoundedBuffer<cv::Mat> gImageBuffer(1);

typedef bp::BitPlanesTrackerPyramid<bp::Homography> TrackerType;

static const char* gWindowName = "BitPlanes";

static void onMouse(int event, int x, int y, int /*flags*/, void*)
{
  if(gStartSelection) {
    gROI.x = std::min(x, gOrigin.x);
    gROI.y = std::min(y, gOrigin.y);
    gROI.width = std::abs(x - gOrigin.x);
    gROI.height = std::abs(y - gOrigin.y);
  }

  switch(event) {
    case cv::EVENT_LBUTTONDOWN:
      gOrigin = cv::Point(x, y);
      gROI = cv::Rect(x,y,0,0);
      gStartSelection = true;
      break;

    case cv::EVENT_LBUTTONUP:
      gStartSelection = false;
      if(gROI.area() > 0)
        gHasTemplate = true;
      break;
  }
}

std::unique_ptr<TrackerType> gTracker;

static void SelectTemplate(cv::VideoCapture& cap)
{
  int k = 0;
  cv::Mat image;
  while(k != 'q' && !gHasTemplate) {
    cap >> image;
    if(image.empty())
      break;

    image.copyTo(gImage);
    if(gStartSelection && gROI.area() > 0) {
      cv::Mat roi(gImage, gROI);
      cv::bitwise_not(roi, roi);
    }

    cv::imshow(gWindowName, gImage);
    k = 0xff & cv::waitKey(5);
  }

  if(gHasTemplate) {
    bp::AlgorithmParameters params;
    params.num_levels = 2;
    params.max_iterations = 50;
    params.subsampling = 2;
    gTracker.reset(new TrackerType(params));

    cv::cvtColor(gImage, image, cv::COLOR_BGR2GRAY);
    gTracker->setTemplate(image, gROI);
    Info("got template\n");
  }
}

volatile bool doStop = false;

typedef std::pair<cv::Mat, bp::Result> TrackerResult;
typedef BoundedBuffer<TrackerResult> ResultBuffer;

static ResultBuffer result_buffer(1);

static void TrackerThread()
{
  cv::Mat image;
  bp::Matrix33f tform(bp::Matrix33f::Identity());
  while( !doStop )
  {
    if(gImageBuffer.pop(&image)) {
      auto result = gTracker->track(image, tform);
      result_buffer.push(std::make_pair(image, result));
    }
  }
}

static void DataThread(cv::VideoCapture& cap)
{
  cv::Mat image, gray_image;
  while(!doStop) {
    cap >> image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    gImageBuffer.push(gray_image);
  }
}

static void DisplayThread()
{
  TrackerResult result;
  while(!doStop)
  {
    if(result_buffer.pop(&result)) {
      printf("got result\n");

      cv::imshow(gWindowName, result.first);
    }

    int k = cv::waitKey(5) & 0xff;
    if(k == 'q')
      doStop = true;
  }
}

static bool StartTracking(cv::VideoCapture& cap)
{
  printf("tracking\n");

  doStop = false;
  std::thread data_thread(DataThread, cap);
  std::thread tracker_thread(TrackerThread);
  std::thread display_thread(DisplayThread);

  tracker_thread.join();
  display_thread.join();
  data_thread.join();

  return true;
}


int main()
{
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
    Fatal("could not open camera");

  cv::namedWindow(gWindowName, cv::WINDOW_AUTOSIZE);
  cv::setMouseCallback(gWindowName, onMouse, NULL);

  while(true) {
    if(!gHasTemplate) {
      Info("Selecting template\n");
      SelectTemplate(cap);
    } else {
      if(StartTracking(cap))
        break;
    }
  }

  return 0;
}

template <typename T> inline
BoundedBuffer<T>::BoundedBuffer(size_type capacity) :
    _unread(0), _container(capacity) {}

template <typename T> inline
void BoundedBuffer<T>::push(param_type item)
{
  std::unique_lock<std::mutex> lock(_mutex);
  _cond_not_full.wait(lock, [=] { return _unread < _container.capacity(); });
  _container.push_front(std::move(item));
  ++_unread;
  lock.unlock();
  _cond_not_empty.notify_one();
}

template <typename T> inline
bool BoundedBuffer<T>::pop(value_type* item, int wait_time_ms)
{
  std::unique_lock<std::mutex> lock(_mutex);
  if(_cond_not_empty.wait_for(lock, std::chrono::milliseconds(wait_time_ms),
                              [=] { return _unread > 0; } )) {
    (*item).swap(_container[--_unread]);
    lock.unlock();
    _cond_not_full.notify_one();
    return true;
  } else {
    lock.unlock();
    return false;
  }
}

template <typename T> inline
bool BoundedBuffer<T>::full()
{
  if(_mutex.try_lock()) {
    bool ret = _container.full();
    _mutex.unlock();
    return ret;
  }

  return false;
}

template <typename T> inline
int BoundedBuffer<T>::size()
{
  int ret = -1;
  if(_mutex.try_lock()) {
    ret = static_cast<int>(_container.size());
    _mutex.unlock();
  }

  return ret;
}
#else

int main() {}
#endif

