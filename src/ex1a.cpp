#include "ex1a.h"

using namespace std;

int ex1a(const int argc, const char** argv) {
    cout << "Single variable linear regression..." << endl;

    MatrixXd data = parseCsv("data/ex1a.csv");

    int size = data.rows();

    VectorXd x(size);
    x << data.leftCols(1);

    MatrixXd X(data.rows(), 2);
    X << VectorXd::Ones(size), x;

    VectorXd y(size);
    y << data.rightCols(1);

    cout << "X = " << endl << X << endl;

    cout << "y = " << y.transpose() << endl;

    VectorXd theta(2);
    theta << 0, 0;

    double initialCost = computeCost(X, y, theta, size);
    cout << "Initial cost = " << initialCost << endl;

    int iterations = 1500;
    MatrixXd thetaHistory(iterations, theta.size());

    gradientDescent(X, y, theta, 0.01, iterations, size, thetaHistory);

    cout << "Theta after gradient descent = " << theta.transpose() << endl;
    cout << "Final cost = " << computeCost(X, y, theta, size) << endl;

    VectorXd finalY(size);
    finalY = X * theta;

    drawLinearRegressionChart(size, x, y, finalY);

    MatrixXd thetaMinMax(2, theta.size());
    thetaMinMax.row(0) = thetaHistory.colwise().minCoeff();
    thetaMinMax.row(1) = thetaHistory.colwise().maxCoeff();

    cout << "Theta min / max =" << endl << thetaMinMax << endl;

    std::vector<std::vector<double>> a, b, c;

    for (int i = 0; i < 101; i++) {
        std::vector<double> a_row, b_row, c_row;
        for (int j = 0; j < 101; j++) {
            VectorXd t(2);
            t << -10 + i * 0.2f, -1.0f + j * 0.05f;
            float cost = computeCost(X, y, t, size);
            a_row.push_back(t(0));
            b_row.push_back(t(1));
            c_row.push_back(cost);
        }
        a.push_back(a_row);
        b.push_back(b_row);
        c.push_back(c_row);
    }

    vtkNew<vtkNamedColors> colors;

    vtkNew<vtkChartXYZ> chart;
    chart->SetGeometry(vtkRectf(10.0, 10.0, 800, 600));

    vtkNew<vtkPlotSurface> plot;

    vtkNew<vtkContextView> view;
    view->GetRenderer()->SetBackground(colors->GetColor3d("Silver").GetData());
    view->GetRenderWindow()->SetSize(800, 600);
    view->GetScene()->AddItem(chart);

    // Create a surface
    vtkNew<vtkTable> table;
    vtkIdType numPoints = 101;
    for (vtkIdType i = 0; i < numPoints; ++i)
    {
        vtkNew<vtkFloatArray> arr;
        table->AddColumn(arr);
    }

    table->SetNumberOfRows(static_cast<vtkIdType>(numPoints));
    for (vtkIdType i = 0; i < numPoints; ++i)
    {
        for (vtkIdType j = 0; j < numPoints; ++j)
        {
            VectorXd t(2);
            t << -10 + i * 0.2f, -1.0f + j * 0.05f;
            float cost = computeCost(X, y, t, size);
            table->SetValue(i, j, cost);
        }
    }

// Set up the surface plot we wish to visualize and add it to the chart.
    plot->SetXRange(-10, 10.0);
    plot->SetYRange(-2, 2);
    plot->SetInputData(table, "\xcf\xb4\xe2\x82\x8d\xe2\x82\x80\xe2\x82\x8e", "\xcf\xb4\xe2\x82\x8d\xe2\x82\x81\xe2\x82\x8e",
                       "J(\xcf\xb4)");
    plot->GetPen()->SetColorF(colors->GetColor3d("Tomato").GetData());
    chart->SetScaleBoxWithPlot(false);
    chart->SetMargins(vtkVector4i(80, 80, 80, 80));
    chart->AddPlot(plot);

    view->GetRenderWindow()->SetMultiSamples(0);
    view->GetInteractor()->Initialize();
    view->GetRenderWindow()->Render();
    view->GetRenderer()->GetActors()->Print(cout);

    chart->GetAxesTextProperty()->SetFontFamily(VTK_FONT_FILE);
    chart->GetAxesTextProperty()->SetFontFile("fonts/DejaVuSans.ttf");
    chart->GetAxesTextProperty()->SetFontSize(32);
    chart->SetXAxisLabel("\xcf\xb4\xe2\x82\x8d\xe2\x82\x80\xe2\x82\x8e");
    chart->SetYAxisLabel("\xcf\xb4\xe2\x82\x8d\xe2\x82\x81\xe2\x82\x8e");
    chart->SetZAxisLabel("J(\xcf\xb4)");
    chart->SetEnsureOuterEdgeAxisLabelling(true);

    view->GetInteractor()->Start();

    return 0;
}

void drawLinearRegressionChart(int size, const VectorXd &x, const VectorXd &y, const VectorXd &finalY) {
    vtkSmartPointer<vtkContextView> view = vtkSmartPointer<vtkContextView>::New();
    view->GetRenderer()->SetBackground(1.0, 1.0, 1.0);
    view->GetRenderWindow()->SetSize(800, 600);
    view->GetRenderWindow()->SetWindowName("Linear Regression");

    vtkSmartPointer<vtkChartXY> chart = vtkSmartPointer<vtkChartXY>::New();
    view->GetScene()->AddItem(chart);
    chart->SetShowLegend(true);
    chart->GetLegend()->GetLabelProperties()->SetFontSize(24);
    chart->GetAxis(vtkAxis::BOTTOM)->SetTitle("Population of city in 10,000s");
    chart->GetAxis(vtkAxis::LEFT)->SetTitle("Profit in $10,000s");
    chart->GetAxis(vtkAxis::BOTTOM)->GetLabelProperties()->SetFontSize(20);
    chart->GetAxis(vtkAxis::BOTTOM)->GetTitleProperties()->SetFontSize(24);
    chart->GetAxis(vtkAxis::LEFT)->GetLabelProperties()->SetFontSize(20);
    chart->GetAxis(vtkAxis::LEFT)->GetTitleProperties()->SetFontSize(24);
    chart->GetTitleProperties()->SetFontSize(32);
    vtkSmartPointer<vtkTable> table = vtkSmartPointer<vtkTable>::New();

    vtkSmartPointer<vtkFloatArray> arrX = vtkSmartPointer<vtkFloatArray>::New();
    arrX->SetName("x");
    table->AddColumn(arrX);

    vtkSmartPointer<vtkFloatArray> arrY = vtkSmartPointer<vtkFloatArray>::New();
    arrY->SetName(""); // no label - don't show in legend!
    table->AddColumn(arrY);

    vtkSmartPointer<vtkFloatArray> arrFinalY = vtkSmartPointer<vtkFloatArray>::New();
    arrFinalY->SetName("line of best fit");
    table->AddColumn(arrFinalY);

    table->SetNumberOfRows(size);
    for (int i = 0; i < size; ++i)
    {
        table->SetValue(i, 0, x(i));
        table->SetValue(i, 1, y(i));
        table->SetValue(i, 2, finalY(i));
    }

    vtkPlot *points = chart->AddPlot(vtkChart::POINTS);
    points->SetInputData(table, 0, 1);
    points->SetColor(0, 0, 0, 255);
    points->SetWidth(5.0);
    dynamic_cast<vtkPlotPoints*>(points)->SetMarkerStyle(vtkPlotPoints::CROSS);

    vtkPlot *lr = chart->AddPlot(vtkChart::LINE);
    lr->SetInputData(table, 0, 2);
    lr->SetColor(0, 255, 0, 255);
    lr->SetWidth(5.0);

    //Finally render the scene
    view->GetRenderWindow()->SetMultiSamples(0);
    view->GetRenderWindow()->Render();
    view->GetRenderWindow()->SetWindowName("Linear Regression"); // has to be after Render!
    view->GetInteractor()->Initialize();
    view->GetInteractor()->Start();
}

