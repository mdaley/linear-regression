#ifndef ML_EX1A_H
#define ML_EX1A_H

#include "matplotlibcpp.h"
#include "utils.h"

#include <vtkSmartPointer.h>

#include <vtkChartXY.h>
#include <vtkChartLegend.h>
#include <vtkAxis.h>
#include <vtkTextProperty.h>
#include <vtkContextScene.h>
#include <vtkContextView.h>
#include <vtkFloatArray.h>
#include <vtkPlotPoints.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkTable.h>
#include <vtkNamedColors.h>
#include <vtkPlotSurface.h>
#include <vtkPen.h>
#include <vtkContextMouseEvent.h>

#include "vtkBetterChartXYZ.h"

int ex1a(const int argc, const char** argv);

void drawLinearRegressionChart(int size, const VectorXd &x, const VectorXd &y, const VectorXd &finalY);

#endif //ML_EX1A_H
