
<template>
  <div>
    <highcharts :constructor-type="'chart'" :options="chartOptions"></highcharts>
    <div>
      <button type="button" class="btn btn-primary" @click="updateDirectionChart">Refresh</button>
    </div>
  </div>
</template>

<script>
import { ChartEventBus } from "./../plugins/chart_event_bus.js";
import Highcharts from "highcharts";

let series_idxs = {}
let direction_chart_idx = 0

export default {
  data() {
    return {
      chartOptions: {
        title: {
          text: "Direction"
        },
        series: [
        ]
      }
    };
  },
  methods: {
    updateDirectionChart: async function(event) {
      let chart = Highcharts.charts[direction_chart_idx]

      const data = (await this.$axios.get("/directions")).data

      let names = data.names;
      let directions = data.directions;
      let prices = data.prices;


      if (!series_idxs.hasOwnProperty("direction_unknown")) {
        let direction_colors = {
          up: "green",
          down: "red",
          unknown: "grey"
        }

        Object.entries(directions).forEach(([direction_key, direction_data]) => {
          let series_idx = chart.series.length
          series_idxs["direction_" + direction_key] = series_idx

          chart.addSeries({
            type: 'scatter',
            data: direction_data,
            name: "Dir " + direction_key,
            animation: false,
            marker: {
              fillColor: direction_colors[direction_key],
              symbol: 'square'
            }
          })
        })
      } else {

        Object.entries(directions).forEach(([direction_key, direction_data]) => {
          let series_idx = series_idxs["direction_" + direction_key]
          chart.series[series_idx].setData(direction_data, true)
        })
      }

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
            },
            type: 'line'
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
