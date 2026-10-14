"""Propagation contracts and linear convolution; no noise or receiver effects."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy.signal import fftconvolve
from .config import Scenario

C0 = 299792458.
DELAY_HALF = 96


@dataclass
class LinkChannel:
    tx_id: str
    rx_id: str
    coefficients: np.ndarray  # [rx_port,path], includes carrier phase and antennas
    delays_s: np.ndarray     # [path] or [rx_port,path], absolute seconds
    geometry: dict
    provenance: dict

    def __post_init__(self):
        self.coefficients = np.asarray(self.coefficients,dtype=np.complex128)
        self.delays_s = np.asarray(self.delays_s,dtype=float)
        if self.coefficients.ndim != 2:
            raise ValueError("channel coefficients require [rx_port,path]")
        if self.delays_s.shape not in ((self.coefficients.shape[1],),self.coefficients.shape):
            raise ValueError("delay shape must match paths or coefficients")
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.delays_s).all() or np.any(self.delays_s<0):
            raise ValueError("invalid channel values")


class PropagationBackend(Protocol):
    def solve(self, scenario: Scenario) -> list[LinkChannel]: ...


def rotation(angles):
    """Rz(yaw) Ry(pitch) Rx(roll), matching Sionna orientation convention."""
    a,b,c=angles
    rz=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
    ry=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
    rx=np.array([[1,0,0],[0,np.cos(c),-np.sin(c)],[0,np.sin(c),np.cos(c)]])
    return rz@ry@rx


def element_offsets_local(rx, fc):
    """Sionna column-major ordering: channel = col*num_rows + row.

    Local array lies in yz, centered on device. Columns increase in +y,
    rows increase in -z. The nominal front normal is +x.
    """
    a=rx.antenna
    if a.geometry_type=="cross":
        # Horizontal arm first, increasing local y; vertical arm next, decreasing z.
        # Even arm counts put the intersection between elements: 8+8 = 16 unique ports.
        h=(np.arange(a.horizontal_elements)-(a.horizontal_elements-1)/2)*a.horizontal_spacing_wavelengths*C0/fc
        v=((a.vertical_elements-1)/2-np.arange(a.vertical_elements))*a.vertical_spacing_wavelengths*C0/fc
        return np.concatenate([
            np.stack([np.zeros_like(h),h,np.zeros_like(h)],axis=1),
            np.stack([np.zeros_like(v),np.zeros_like(v),v],axis=1)],axis=0)
    row=np.tile(np.arange(a.num_rows),a.num_cols)
    col=np.repeat(np.arange(a.num_cols),a.num_rows)
    y=(col-(a.num_cols-1)/2)*a.horizontal_spacing_wavelengths*C0/fc
    z=((a.num_rows-1)/2-row)*a.vertical_spacing_wavelengths*C0/fc
    return np.stack([np.zeros_like(y),y,z],axis=1)


def element_offsets(rx, fc):
    return element_offsets_local(rx,fc)@rotation(rx.orientation_rad).T


class FreeSpaceBackend:
    def solve(self, scenario):
        links=[]
        for tx in scenario.emitters:
            # 逐源载频：异载频源（窗外强源）必须用自身波长算 FSPL 与阵元相位，
            # 否则 5.8 GHz 场景里的 1090 MHz 源会被算错 14.5 dB。
            fc_tx=float(tx.carrier_hz) if tx.carrier_hz else scenario.carrier_frequency_hz
            wavelength=C0/fc_tx
            for rx in scenario.receivers:
                vector=np.array(tx.position_m)-np.array(rx.position_m)
                distance=np.linalg.norm(vector)
                direction=vector/distance
                # Narrowband synthetic array: common delay, per-element carrier phase.
                path_length=distance-element_offsets(rx,fc_tx)@direction
                gain=wavelength/(4*np.pi*distance)*np.exp(-2j*np.pi*path_length/wavelength)
                # Co-polarized ideal isotropic scalar reference, no cross-polarization.
                links.append(LinkChannel(tx.id,rx.id,gain[:,None],np.array([distance/C0]),
                    {"arrival_azimuth_rad":[float(np.arctan2(vector[1],vector[0]))],
                     "arrival_zenith_rad":[float(np.arccos(direction[2]))]},
                    {"backend":"free_space","antenna_model":"scalar co-polar isotropic synthetic array"}))
        return links


class TappedBackend:
    def solve(self, scenario):
        p=scenario.propagation
        gains=np.array(p.tap_gains_real)+1j*np.array(p.tap_gains_imag)
        # Prescribed absolute baseband taps; identical channel on every RX port.
        return [LinkChannel(tx.id,rx.id,np.tile(gains,(rx.antenna.elements,1)),
                np.array(p.tap_delays_s),{},{"backend":"tapped","spatial_model":"identical_ports"})
                for tx in scenario.emitters for rx in scenario.receivers]


def apply_channel(x, start_index, link, fs, num_samples, output_start=0):
    """Windowed-sinc fractional delays, then linear FFT convolution and crop.

    Input must include the required history and DELAY_HALF samples future guard.
    This preserves absolute delays and does not wrap the end into the start.
    """
    half=DELAY_HALF
    delays=np.broadcast_to(link.delays_s,link.coefficients.shape)*fs
    result=np.zeros((link.coefficients.shape[0],num_samples),complex)
    if link.coefficients.shape[1] == 0:
        return result
    if np.max(delays,initial=0) > fs*.001:
        raise ValueError("v1 supports path delays up to 1 ms")
    if start_index > output_start-int(np.ceil(np.max(delays,initial=0)))-half:
        raise ValueError("insufficient input history")
    if start_index+len(x) < output_start+num_samples+half:
        raise ValueError("insufficient future guard")
    taps_len=int(np.floor(np.max(delays)))+2*half+1
    for m in range(result.shape[0]):
        h=np.zeros(taps_len,complex)
        for a,d in zip(link.coefficients[m],delays[m]):
            integer=int(np.floor(d))
            k=np.arange(-half,half+1)
            taps=np.sinc(k-(d-integer))*np.kaiser(2*half+1,8.6)
            taps/=taps.sum()
            h[integer:integer+len(taps)]+=a*taps
        y=fftconvolve(x,h,mode="full")
        begin=output_start-start_index+half
        result[m]=y[begin:begin+num_samples]
    return result
