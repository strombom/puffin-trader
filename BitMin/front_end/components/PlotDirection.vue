
<template>
  <div>
    <highcharts :constructor-type="'chart'" :options="chartOptions"></highcharts>
    <div>
      <button type="button" class="btn btn-primary" @click="updateChart">Refresh</button>
    </div>
  </div>
</template>

<script>
import { ChartEventBus } from "./../plugins/chart_event_bus.js";
import Highcharts from "highcharts";

let series_idxs = {}

export default {
  data() {
    return {
      chartOptions: {
        title: {
          text: "Direction"
        },
        chart: {
          type: 'line'
        },
        series: [
        ]
      }
    };
  },
  methods: {
    updateChart: async function(event) {
      let chart = Highcharts.charts[0]

      const data = (await this.$axios.get("/directions")).data

      let names = data.names;
      let directions = data.directions;
      let prices = data.prices;

      Object.entries(names).forEach(([exchange_key, exchange_name]) => {
        if (!series_idxs.hasOwnProperty(exchange_key)) {
          let series_idx = chart.series.length
          series_idxs[exchange_key] = series_idx

          chart.addSeries({
            data: prices[exchange_key],
            name: names[exchange_key],
            animation: false,
            marker: {
              enabled: false
            }
          })

        } else {
          let series_idx = series_idxs[exchange_key]
          chart.series[series_idx].setData(prices[exchange_key], true)
        }
      })

      Highcharts.charts.forEach((chart, chart_index) => {
        console.log(chart)
        console.log(chart_index)
        console.log("...")
      })
    }
  }
}
</script>

<style>
.red {
  color: red;
}
</style>
