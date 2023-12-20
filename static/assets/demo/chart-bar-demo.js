// Set new default font family and font color to mimic Bootstrap's default styling
Chart.defaults.global.defaultFontFamily = '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#292b2c';

// Bar Chart Example
var ctx = document.getElementById("myBarChart");

// Fetch API를 사용하여 데이터 가져오기
fetch("/api/day5-data")
  .then(response => response.json())
  .then(data => {
    // 데이터 형식에 따라 차트 설정 업데이트
    var myBarChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.map(entry => entry.day),
        datasets: [{
          label: "Revenue",
          backgroundColor: "rgba(2,117,216,1)",
          borderColor: "rgba(2,117,216,1)",
          data: data.map(entry => entry.total_sales),
        }],
      },
      options: {
        scales: {
          xAxes: [{
            gridLines: {
              display: false
            },
            ticks: {
              maxTicksLimit: 5
            }
          }],
          yAxes: [{
            ticks: {
              min: 0,
              max: Math.max(...data.map(entry => entry.total_sales)),
              maxTicksLimit: 5
            },
            gridLines: {
              display: true
            }
          }],
        },
        legend: {
          display: false
        }
      }
    });
  })
  .catch(error => console.error('Error fetching sales data:', error));
