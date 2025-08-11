package com.richaa2.arpdp.kotlin.common

sealed class MeasurementMode {
    data object Camera : MeasurementMode()
    data object TwoPoints : MeasurementMode()
    data object SeveralPoints: MeasurementMode()
}