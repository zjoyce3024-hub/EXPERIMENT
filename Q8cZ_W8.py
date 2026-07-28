#!/usr/bin/env python3
"""Compare two parameter strategies using embedded training/validation losses.

The script is self-contained: all runs needed for the figures are embedded
below. Running it creates:

    train_loss_fixed_parameter_Thm_1_parameter.pdf
    validation_loss_fixed_parameter_Thm_1_parameter.pdf

Dependencies: numpy and matplotlib.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import tempfile
import zlib
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "parameter_comparison_matplotlib_cache"),
)

try:
    import matplotlib
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency. Install it with: pip install numpy matplotlib"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_LOSS_INDEX = 0
VALIDATION_LOSS_INDEX = 1
GROUPS = ("Thm 1 parameter", "fixed parameter")
GROUP_SLICES = {
    "Thm 1 parameter": slice(0, 4),
    "fixed parameter": slice(4, 14),
}
COLORS = {
    "Thm 1 parameter": "#0072B2",
    "fixed parameter": "#E69F00",
}
MARKERS = {
    "Thm 1 parameter": "o",
    "fixed parameter": "s",
}

# Layout: [metric (train, validation), run (4 theorem + 10 fixed), iteration].
_DATA_SHAPE = (2, 14, 100)
_DATA_DTYPE = np.dtype("<f8")
_DATA_SHA256 = "ba7517319ca5537c8b823c125b19ae6e37ab1b51ce347a21cbdde4a15b8d2c95"
_DATA_B85 = (
    "c-n1ScYIH0`#$#Gn}lPl8sW9)QPrwe?a`__rCu$yszqxZtFIQds%V9SS`7)YXHIM}V#SKcu_GagEL6g;?(4q4pFZbTf8_a`^L@^9"
    "o^g-sy6!{MUjM6P{Qbx1(Uw{(PnbVl-#<vL)2*%ky0wEorE6>Q%ZhaU&8O6weJav(PN|!CF5aK+Ynx)w@9P89T7jR^I95O)_19*#"
    "p=+n*0ea3X-J0&}mHVTA=6=k}FN=QHt4%_`Hdz(u`TQ}s-`=0D-S;Myr}6q8C`;|`NhPVRcC#3@?;k8m{o-yEqUWl|<frkP`{$$g"
    "EN-2f?)S*bMfdyP$ZY%iJ<le*dfi%=CdR#P-JA#Rzi!Pcf$6VX{l)SOx?Z12znkM*#lLQyO8Za0ZVe~!9JR#)@6qpd{T{t;?H#SI"
    "Qv2vd42|1l(h>T7-HQ;qe>3wk-A|0h@2!^aqxSIrE7Y%f|4a0q&;2h_n>cYD{a$oZ6uqxX)*m!Z?@4!Po<Dx_kk+$%`9|tL>yu^F"
    "FJdB%rw{T^rRU!`K;xRjVs6uYt<_vw{|5hXdVcGKd34=xIjz&mk9Apf12Lc7En{ii9DekiGb-~MUEjPhmfEE^W>Np4F^A}VO_$H6"
    "_mrDBk;d1v?$Z4_foaq~*MXVTf5CzCw4S%Ju-@?gw6E^sz+3cu)8!NB_hEiCFFPm;`#!@zh1Mg_hxeaP{oigijK=Ng_dDNDrg0Bq"
    "fA#$prqFye2X_4KN@|T_ul%UJ0sk=9w!!by;SY8X_=i~o_q7ZC;Wvjrq1OHE8r^p~Hieyk*M(ZQU@_d+R^$0CACiBVE9T<4_RZ+N"
    "-eEfY<OKXtn_r9OV^^9CKS=(S?%Q!W>At(XU^!~r<t{^Q!@tNcwDo(6(>mU`OMYP&`miv~r(*%Mfqe_o?|c2oPxL0y+30#jJB^<2"
    "`{X(8SL_k8XCrMi?VmBT%wM#xWn4NZ<6;o)r?Fu+`Im9-$SFFH(Sa9XPwkG<@15e{S1(LDcWv=ZI!A5aJFu^_<+su~<(_t(`VIL0"
    "2Axx%(eT4A$M533C&&)9f+NT-%!e5?KfS;!dd``?>?XAf`n%Nc+;;L?=YDUz@3WQ<=si;-7t(c?D&!aXu8}mZ-go^I>VJ9%*{{AX"
    "=ojiYcRTru88wR5rO)|=?9BWzh3;EZvys1B6E>57ILRewzE1r}*xjU$f1uyD<UU5v{dAu8%`VZE#<5zbko{UsdeZ!~ZJiO1{)i<%"
    "H-73%@3G@6(|tQBmF(B*vyuGH4LyrE(6|epuRjKD#Rl-7gjy8G^y}9XsP!8|`(vG2jX3mO63xdsvVIQz-v7=u_=iSz>P|-9(Ao_k"
    "`*bUzwX?l~-xQdF{E&JO_T6A5?DU<L=s&A1;`tx&FY^^YUr&SoJVPGRZ%sj-y7~(5yNq$P$XNK(`AYP>bFvcpMc{Wm0@wEGmGrxH"
    "3G=bS!YFT82?NMa^n}>bbe+7C{KOqRfc(V#&nx)JkqMMPwDBd2(EZKD^HRSp@8+cEr!Ap;p>6E+0`_+?k@m0oUh)h3{gF8L@f|6y"
    ">!pW~eLJPgBY*f`rF}PwXOTa+87X9MdXu-Pzd5wLL%-)tgk3$%OXF!5nvi`N6$VlK*ISOJ^RcV$41v85qw{w^oOTa(Ux~)Ia@`}l"
    "wo8s%NzZlcx|-TzXQ?#<G&+Ct(Ije}^0nY!1J@&Nj*NsI*!{?_?1_QY&o0pRGVG@u*|%A97vgcdhR"
    "8#+=KMszZ|oco`^5b0Dj(7M%!BzUPG}*^5I-8XqxEZpkQdF_y*E(*ea&fHXZ3N)kJ^D^<d<5#j+keT(zJh8%Z?OR^i$1Y*Q+j#r*"
    "VsZMgHn;EQI};^CjgGGg}bdx5F00ZxhqWZ_PG|7pOf@o8D)&FTH@CySEN`>q-XA$I2RS!9R4^dFiW&^L=&TnqxVDfA+<ZJ!;jO<9"
    "bwa<bfk)(67O7c&@M?^2vX0(|yy`a6PXo`p3<HJqIM9f6JfY53wP@aih8*k6hk@e&096_$PlsK3lsJd8F+U;H$nF<QJ|p9C@SMF!"
    ";su_T(2<siou(&iO~=2hM{P1<3wge`=jv^(fxEAAXaE#_j)-@`m}_O|*-?ptx#p?t?fqKP?gaTZhic38@DBk^56RFS|)9*{#u|1L"
    "Evg&*?n0Tbn2k*!oKNO-%S6YX9oGgYN%+{V81s?oOiXmxJkfw`a|0y5I01&NtypI)5j+HS)sxoP-;+FFMn?nqT}3KPwnS{j{(S^t"
    "-dG5A0$@N+8YmUeyR1ciE(+$X6lAOL_E%z-fmGN4Y;urE7iQ6^gsgop_47ZkI=}#|m#D&+O_3-1u-3#Rnt#)FNv0p8$ReUr6}K7}"
    "uYE*S{PA9JV;_0{s3B>TmB_gE${i@(0-I!nJs>Klb<O&y>gX@HO;aqjD!YAMNZ%<EZ~!tiuk&dz_kwf$Oq*!5?dOKwjSW2>aM0jQ"
    "Uw`e-}pAYrdvDZ{Jx@`>#!({)GJlc0WEAb{Y36>~dclwDb1D&bwTOpSV}xPtz`womx%zP-~wYSq6EcNh!K6e!T=ezvWZH|8B)D)E"
    "b9Q5U$bl-zD6y<^GKDjZ<?4#c?y>6vc6)x|^N)eK{uw{obq;*{xlBBb}FaDuP-&qcQE1Su_l|{%A9bXXepLk7@mV>mi@U_Cx<pQ|"
    "KJ5Gr7<&=trFIfcW#+muX4BQJ=71Y@p}uO@nCv^%}zoAJ~&!iigJ8e8d~vQa0J0wd5}1@wl15F~|MrcYRgsZS<a|ODP`fjTY>t=j"
    "ZIA);zH`l&;^&L3zsgtr^aD-oabIDO2d$+?<T{2BwmonmISoyq&5o$q&sVwP-wR{b|^1*Jz64&i57JzcX_YPjQyC0?$Z1OY?L`k3"
    "C8A{%LD4wawpzAD%Bx{%MAe0#2Gbmi)&GX-xUX*nWxbTP@BZ4iuUeN$s1Bw$O92Zvgi_>_GkOH7#x-AMdeoj(%5w%eDajRXC0Kv9"
    "nhc_|30~BOwzgU+Ps`ATLKCkDJTZlAjs%R$l=>J5Ja7!;|25S2|(dMdl*jbstLe)`pi^3mi6u*6GatmCoH7F^Kj@E70#c{KG{2o^"
    "c3%F!Ubcd*wag(P8&0Al@SGn}1Fq9An1}AwRHgEG7J6N8h5jZIp;1`!{bCL_BX9NqpHDw1uv<ki+zxwx%5Uk9qYrJ?}obkR"
    "8u=M?7EjJK-B6{1w@YK5jMfW@qUJ@@uz!cfvvDwOq*mGY-*utU^KQ;0wUpPUs0bf2YHC>`!0`<bn6Mli%5a4}e!%9-{kpwf)cN`T"
    "B<_{yCXx4}g0T$WNT@eiZMu|K>n`I6fOVscI<Y6>DbADBy%~d#T;f8u7dd?8Tn+lAd$xb!MDqkzHxqH-JyfSWD+`FN(Skp0bag(="
    "!t>?(gRj|0=ekd6};k5#P4-1?%blv3#^{z4b$i5ANWrv>$f!HH71}_A}w1vqxavLs|g;<eg6I*B6X{-5k0A|Cql2Hu(0CKk0hPdt"
    "}#!UmEOt;;PLwk4j(C`?T`!A--pirSbKJvnVcE+27d){!}3e^Za}{&8u}zT8G|Y7vV^K+S{wBt+EaNRP!3*;isd)2fzN{IdI@6{Q"
    "lNh;!(!H@!&;0>fsz)-=W{NPEFuP(YNm-Z`@l#^Z5Qx!dK?!A0U6%DTKIDXA;igMa}eo@^d%kJMfIwOW+4pnNKX~2OJdf6ZrX@oZ"
    "tb8Ybt<G^r%jDJFHh_>i6G5;0M11lV7+mx{*H^{=EqAXk9PD4+i%nK5PYXoAW|CbdAom&rTWeLAS+R;MUIyl0P_W)2P3FsT9^fDd"
    "{2jFj}W!16r^Celh3}0h=-2*)x=1oEUD;EC64gFdV<<9ffh6;@}g0X*5r5#4qG0_6y`Mr%P9yPsA^wIJcsIL1#(B`QHctJ`ZY)eJ"
    ";}oIQlBC?SdI(@6L`8_*?wQAn0Uc$bPM!U1)ypkKd3#8?%GY&~@J~q=&hUdI7(g^Wg_6rI0^H6n_BRJO=o$_af|PjYh!LHPM<?fd"
    "lnLMK;m>H9NM#FQRB1w@)3~XQRXg+DB{IS;A#*P%!ps`gZucUk38Vf*#Npdj1Oh*t8MhAm`nZ$hXZJ(YoA8h)?dm0K}~tqiLUvkJ"
    "Hv*-#Zh3(xMj<9@5$tO+Y@~X(KO;COoJWJ-Y(=xHNb|o+#p3&bTMUXRIsTo&kr}r"
    "8s1jFGBM-R*#}{(%uSX-kt)#sI-;xmHuHP=w-2?;OlLBLPz-<8Izr_Fpx)Dg#%YzYYjbVWgGDF&a9^nhMww7Ye)Fs$f!lQ%FN*cS"
    "6zrjK8U{vUG>ox=&A#A6EC(qrV;+PPM6OI9V`HRAhQ|tpU}@KPZ)7)vuJ(kA7P%S`XH}XJ3+WypOO>2Zb~%edpqk0#RDTTne5ixR"
    "vK~bMjGN??ZadbPFP8bk7kAE!O(}=Y=X{o7I@%Jqfo@_tF&)c(KXwk&qg1`IlKjaSF}F(;>0@O?TfM@kDm+$uBrO)CG4&f^3|tvX"
    "g%7FvGCt)bI9JztkKXTYrKLES!z1?Pj~Pht6@Xnz{DGeY5eZV&{f}o{aC}>6Cbv6hg|@!%SZO@9&ZLbxh|9DufMyG?A0g}NOtOs!"
    "F_8@E$G+He?i{-`XY3K8Q;P`o`!&b?=DH}(%xJT{dqz6Yw+X6=(l?>;bJ#4f&ANAI?aWCoBkAevL|@KfXd+8HI5_yH|(4V-8=tT="
    "npq(9rp70&EPMu$ba4PFGG>{j|0E`FpPe;3y=IA`FH^E=}VqxL|icP!S8A4pO^BMo)J!VqkT95d}ZSM2dH1a%9OY4W3Q;Ss&s+Ap"
    "jpTt%{qZc59oz<$4BrF8-8K@ei%B_@FU<MYjRVo?+XL($om5RQU7ad-Fah+fv>kDe(cUof?xD^P<M#j2i+(#6xWGYXn&od%kn^<Y"
    "EP}ZtPtuF9}h;}@LNuH=62H&ze0v3Lnm5F@!qYS481cV4R}0gDfmOVrWCL22DN|}9DnFBUsVOpNliptdGImqqj|axcwxB(<QMLHB"
    "PniL$&qw^R(Rio^jxq`yu)en9_0sPPr;q=lk&9h`psD6xw6^d-!0!p{+jp&jb{#u25;Q@7<O6XSLm3FuOj|#&kwxyXM^p?6N|uO?"
    "`#Ht|71GiT|b@HYXsDTZhEo^aL&*GI%g|wF?ex$0oYxkKEPSKTZ50?jwAeLk50J?+!RCc*Y>YYaoL=|ipF<N1jFtJPC`7$_TDz&$"
    "TZ;1h2?>lR=o>c>t_O|O-%squcgC}GsZ$U_qz>Tm!3iW%>NxHoM=v5xr^T4YB|>bY!&5Mr{x^@_qFPT&zxR1^3S=I(9<JMBmWor1"
    "99=IT*${=x*~5?KST4>k6xxYWF&S0j=HiF{AO+*%)i2xWY%|~qwcrB&%dk;eD!lU{NmhasDo5p4SlIC>KA&6L&QU@k^>1x>6Hc&z"
    "i=BIBL6T?wnP2lNoC3(R?;oXALh<><R|v_cBnh-(y7$~20~x0^Izfx=E8iyU0-ZYgC5k7_=x@!c$U@rEb?KCP0(q2A`ZJ3hk}osz"
    "jPBi{T%3yYwJV*4y#N1Xq^rQU$94DoGY`Dk3KGd^DLiE@!gnE9PvA?2k_0y0^mQvo1wSg`4;uCWetF<%6|p^IWLmz))*CVg2o+M6"
    "n58Ywu8J?6mf3dCh(VxNU{g3++66Cn*+edbA&+09TQ2sTmO6$@>e$WvvUQ(uI?6~eXuJhP#!e$Ed+j>9RfR@)fMp}X*l_T^UHAPz"
    "1QDI+-+GH_7Uj^T;H-H;_HBlSAlCQZh$=x2n2qajJWj2WaN=WZOQ+vYWf<i!`_d48wT8(7>4-#jSIipz`FA)6ZyPE74WFJtD&=$T"
    "D=<ib`ai|rz-X(_aIuYKIAZP-v92wpPtS@UF2CR@W8h#D1Ml0OHusMXZZnVp1KYEF?up|nXumwm-5Bp{ExJSPWQcY@t^vY*$eeS{"
    "nkzJjeFtHQy1g{pV+z&I%|nzs82-rfv3014!&J)D(Z4;%R<lUI23j_CK0&(?Jmf}muC`wHyf8i-ktj33F4;<-qz$z@T(U?fm6OkU"
    "Cr1&6mg*haIG0x82qFX&fRK}0lb;B1$4yE<|ED?g+A(jcO2(eANr)W2lXL${P*C&e`j!-zh;z0YnMQ)=jApVxA14p$${2<gy)Sc="
    "5x1rJjSnPDz~?}AMd*s$L%?8llZxK?)Q?<=MImX!tF!G#jHQNeBXM^{bfD7c$@_O{T^TQI*be+hv#L<Iv((R5AeK1=e(!R&p{paZ"
    "-&XwZ{+7fxu48)4gda1ov*n?ou3iHzw>*vV7{NEuFt%{<MMmu931A;<{#>P>k?n{`b^$;^P;-XHm}>3b-MgJ{2@w*f1C`*_4}()_"
    "e=T#_g@V`K3UlSd1iMC^xc;8QI~6-LUv_64hGK3o<R2He)SMMq-+FuNaazaZ#k#BA)n}xWPf(l6Ur-Eztg~5dV9i4R;vTx^GnAQK"
    "Gw5(BM+=BMt-D?SxEk7^tuFmUB4Ur<=8jKD+7xn4u7@{_$aC;c=E4XphIPJ#rGP+)1dE;ZH#@HR~Px>Vj8Z)OTz9GzCxXE!*JBMp"
    "FKuA`XHEEr%fH;jO_u4yR$AKpKM4ZyE1amgASZ@7ChqTy2QVYsb5Edf7C_ZJJ1XEQ|V{mkaruCo!U*ilfLYBo=fX@=Jdk6+SLdCI"
    "a?U`>5Y0cjuo7Wy2#}7h?8qaA<w2IMFV$rrTFSrPX-^JkpjK<66%vyiQ>?!rzg?4+Thf6&?n}_0Kb>OK6XqeKX8Bh75IHr2Jq%Xu"
    "J6rUv>tWPbKobxMG>FU{@9K<{_S4i;A_(e59-;6VZEK3?ZfvrXOVxZSAu_rUI&g?_7~#c#8Oen>+^x1LMu?d(|#-!4BeqOz0ZtkM"
    "ES~C`3>^d*UwPLx}Bf!n$|lU`r^7-%h*q$>;C3~e>eCH^}P|<pjY(<p3;WgM*bKu5dKlR0raaN#9Jr*KfqUuTabS^9~J>muaX7bY"
    "VSuy!Sl=E`=_!&sC!I!0Dl-81-=pc26Wug-N4U_RVUn~-AktUV~-jO{8axa^qn7ep+3F05pd>~Z>Tj}H-(;C7&?ooPY0g5P!sWZW"
    "d?Bj&%=RNE(1^53xiSD*!_a)0oHdx;LmUS!~XI%fWItV30!ro0dW7qf@tfPhmMn43^=Zs3w*u$JKQ&*pIN&jU=MY-Qham2seyXj$"
    "J=rKO)dCaFzR5&$5-Kp6I!AkGI<*662UW3ujz*GoSetap+~Jghx*&4a`21gw~<%v^o^)f1S4OMEe|~_e**Adrb%|}t~hfYx@mFf%"
    "0)gzJ?7Ath`YV7VgHKU$GVPn03Z7K`Wg63LL~I35cq%OMAD(GksT0U1`a?y?s*&7_5aFZyl=;%E;y>z3Do_2L#G-M4?TI^Wr~OHp"
    "}{K<=bocZlDT9Rc>Hzv_r?LxJ3943U1M6?$A~*aDDLT39^$<<27`Z%yGuCE99$Q8cFy!=|I8omT6|Y%m0ON{@orD>^g@x~;qkf9@"
    "5LZoSNt!&>lm2>{Jh#m@bU{^lz?8FpZr4~`I5eOvIo@0cTd}XL|e5!_<7)c${S|*)Li&Ja3k<o`<LM31>-4yIBVa5zS80x*_-8`2"
    "K^}hYw+}^@rcj4E<c2Rvw+UQTDKm2F69XFbm0rAADk}=99;Xi<FwCtu`hP_h7{k;`18oenQ7p)O-4aqDY75?obxWe8w}c+4jk*EP"
    "SR)vbfpijrhun{=jq+=p&qy=2L8}|7wN{96%Ab`bq9DrvG-B8%O426r|P4<@Ygle+Wl8hzSG`@zHA*Ek9mLbJ@l@riKxdFr~^Ejn"
    "h*H2@_N)sS4|*4(8tt+9R*CLcx`N14*!_c2YGF(hJAZg7JlTH13Ylrm%#rQP9u)=>J9!A--P_Y+#e0yq0)^t;4fIOQ5N-gt$R-Jo"
    "P522Z|5f<KWw@VylenRS=U#=-$twgA8`8tzh1hB?^1>ake};?4ua<$FN3(Bq@%81qapnJt=iB_p7)0?{K-l>UpH4E^5c8jAIQ6n5"
    "g*45fu3A-%w@)P^?}p0dff80$=6Oj^;!#H9M1iWx{UMcs^i%5dwuoy|2LlJy;dE@wch@9)z*-HwfTMAnzeYG>fEm?k6+pNhaT~FK"
    "R<4ncUWckx+M22$nVe2zyHPfG?VvDa4X|u=PtL*J6y)g+HI9LI9K?&n~a0GzsC3VT)e*#JPzY#i}A1}d_cw*zVVc=S$8p?@qOkQ<"
    "}MXqo4gK#`Hi+u#o<mEzlX=Om^Zjv+p4^42mj7|!s7Q@vM$~iH%{dn4!_sR&HK$f&x%vWaf4L;<nniePKY{=@km{l&HTlc_if;7>"
    "A#8B$LHb-Pvrd+ex`fu(`CCe*bfZx6Shm8@4NC`UG;gx<5yNawZDwxwSU8NU-$Zv$1gN)b@mI5{la9w(5rEOo!`TLp|d~e<@kPSe"
    "xCWbR+Q(VsXScxx$r<k?2P@#PF8ue&OFu@e~E3Y^b<R9nAcm}>gnn^%C*lP4|98jpXY0b=VLup{lj2?(zw67ljkA+AmdBR>lS+xe"
    "r&KEY5cs!_GF1YF+bP0aldp`FR)Y8{m|IYTzRkP61*=u^LgV>z83vM{Fmd7%X*0ZOpP}>>mJsAe%~?vo%cz%Rex|c^R;}pD>_R(H"
    "6OV3J^8_7*S`ExU$tY);}>?_fA<T=^PKJRPw6MmOFl6@t@9?&ueRE1HTgOA550=<kM9Kjo{vBGXMfRHuhKbx*oC;|IBw_U-#O0t>"
    "{-s0?Z(W*<38v8WxsJbo|#Fi-G~n4N`JN=gYDa8JFrA2dgk#*$^Y?ceD&oA_6w8!TjPGZ_~i-p{U+xDJ5se%lk=Rx>vJ3)XD>f5E"
    "&GKn`qh0k4!9i8ZSiN$H?l6@yk(wG)pf}D8r!jcTh-|t&adV+PrjCXA?tkV$wx8{>tgN_9-sYP@Q)YwNd7QA{v-A*abDuScO1imh"
    "is1@39j<tBoD3<|Ip=H{Db|#dV`;@!Q*jjR#EeY$$p_#;Bh2xl;XBHk5`!cv0oUR?=?O@JyX>UHMU#(zKV0~N1k&?Qt^gyL$zzkQ"
    "x>0}`$UbaZZwaZ;fcedKZ?%DziW*9#2+-l7x&fo>AQIxiOalh_lk-G^)%K0eK?Z!QJ3xAX1g?apY-Rdf9dDB{~^_XB;Q)s)x2pY^"
    "8G|rKX(}?*@ENPKkbcb+!vi%=Xm099<es7`z!fba52|2HEqkk=MRbRnwOV&@Q?5bpS?1^aoJ97wo{97zsar3_Nq%fXT0OE-MVE|+"
    "~W!#U^})8@O_T!w$vrqZVis}Mh+DxS%T}+J$OOzgT!zC-HK87Srhx?_nBf>=Tw~T#VO87KEIQy{|G)Y1V<d_=eVCC<33U2zJ6HcD"
    "Z)4G8>*lB{2^846UHsozD(Xn;Wci!irWRx*$37A(WIX8O!ZH}FXESChfb1;w+yyJo9)(q=)t2rUn@byPjY?|w*=ow{*bzp&hHcZu"
    "(+P13y-?1<}Vq?<a&+G{KWLceci*uEssAK9^CITp3!TlxL>QH;u)9uuvJdQGbY=;&U{$xPvWJ-bDiU&;D50X!S&j8^}I}}>&2__$"
    "l!RQpHk~@HeVa3R9;{l<$fHeO-Hpq@e4TzW4G!@Qh$>3l=vw5L)K&Qd0A4Y;&F7gXN~RFxa;AS+|OcuVF}-6ztgf*d$9L<-gi&k2"
    "f>T>cGcd5XA5pIM86PR!|$<osd&*>x0JZSI*0jOtv|Y4$1`rLe(Of@eICcj^zdxq(HxJ2e;9wM`{kbW<U`rVKh^ya{44lb@VyW3c"
    "yN&58eg6f{LlHq5uJ$hgU&d{k$SZ7-y(d?>vmqLIMY{;mG55Io^3gAUaxzV*B!6wGmeats>WZ5W45O1BiaKsp9=pKo-Fyq^u!%Y;"
    "$(svSAFLuy3$iMev7?}4(8+2`FLJz7aH58y;aqRguh6gS@OSgMy(G^+|)$xx~0Z#Gg-wY7LR"
    "8{t8rfPuf}oQW_)EoRpY(nOYMcHPJcm-D|U!y-><5AoF?BRa2#+%??_N_n8y6ZNLF#L=wgBwrQWZJj&Mb_d&wKdbq^m(^~6~ZkM`"
    "=eKK-bUhcA2dRKfplWmQjgIDgpEvfr5OM=tYYLvWPj7uJ!CqN?7i6;|&HPmu3E#1CD$W;|x)QOC9TcgZs*^K#*#62COIU!C78eAk"
    "Ejxu1_m^0~Q^r{X;E{=SD-NPLrd@cG%P9-J*aM&fs*=lhEUHQz~mmb~fHeVCVPyVU#oUG=*Vi}`{jb|^UQnW|gcVQL%~elF`aBya"
    "Ck^OEEl-Qy?bV>KU(zstM?ZwfwjUwU-zjjB#9`9RyDeplj)AG}YZr#rUiI~@<d);#>%@#sjBPb|@kYJ2jC=&Bx`EI7*&zbNPNhw>"
    "`#5Zq<5J=;=;&{X{I(@O<^#e2S2N>%sOmwyGnr+IXpD=Lo_JC}H6C#vyH;-s9P4?hS_$*tPG?s~pEI;5VHPd|R4;y%e&f&=v9YCa"
    "SEr9V*lgO3M4@#F>ZAJ#p!oa*;v7Tdk>TFE<7$FSo({G8`&q^bPN7F={w{k}@-8wTfJ{gLPUF2+^%9&Y2+@1o2UkKUZ2eqZM9S8="
    "8;JSI%lPqbJMt`fg`sOAr+j><b+k521&{6qMM!8(%3JjC|Y;{<nE>>sAT%FBh1>zqd{&LcX<bCdIi&GBC9Nmh1GT$gi^I)(7)$Lh"
    "Wb?lC>z=lQN(Ph6IIMwAB+yj1IjzWT;}Rd15`>(l*(r(W{t1c_=K_u-RgDvlSPFW)zc{YiaMaJO?+tsB~Bxn;j{j;Q!lc)w2{JEr"
    "npO?U+7=l{o-MQ0Fw<fcbgjZyKA=(7@ce0V=p#h=1Er0(a#&vL%WYJ3sB);I6J)O`^@lsG5nB7976rVqD!byrUv!tm6^y?$bQ>)o"
    "|ge=%yQ*B0Y2L+W_d)p~^~t@v3rRS%Q=WJz74^1th3zP5^r`z&1@&yj1chgjmLB~;zOV4Nm#(f56>;EjB$ekQ*QJ~Cp|d?$V>*M^"
    "60`p(Cr1GuMD`}M{1EH!`WM^)bLtG^|9@|)<NxBs1IdF%DQeC5#ze03Y)i#I%W7pK-)+(fm0A^RcU_sV+YzVJz@%Sv7p{XxbT{n="
    "%FAoWtggHkW`={yHiTx{P`^(@=NJA68oN6+%r4TYC@>T~uRs=n*hvCK-UjwRo97*dxgt$yd>6jOCwGq>7`zT`^0<a74b+0s<~TyV"
    "4SW3dCl`4YdRZXovTt8-+i@!F@;Kl0$_M{2$n9OmPRJRj5Zy|>^zjqP8P?=B)db&4k*efE~Bue#^d@$FbupAa2K^bD~p!8u~LG9U"
    "Yricduc&^&ca@n^9!eZRU7w(uFj|9d?A>Y1viOa7Mg6x~pC0NHmpM$J!R_dfr5?s;#VN5>LfafkZ5@4FdKJ=<H)b3OcA@`v@NnlF"
    "5HkLqgtH#t7*(z1UT!qfd!zAil779Q=vS5oH?ye#=%;+O9`sT8$!J^HZZW2t-T9=~zC{OzgdJEUwXFA<&6hl8cgE_&NF^_&Ff%6w"
    ")1KAs!p!O6)U9?tp0I<9_ip)ro~<vY$J_Dhd_#HjPs<a?GxmB-8XWrE*DAMn-d1$UiL>-AFS^wq_AzIM1LZ$4FVrr=_q{oeG<JK0"
    "mOx~s-};dd_klPh>{ugdo%F6uYcJS_3il6c|G8(uxj^L>cz$s0Z#RbAyDf_KC(oC+#GmpX{e`9hQY;i+fa+=@MlUr0WXI<50U<sZ"
    "W1B@Rm6!xH``ct4wZo)YJM`k~<W7!R&sJSE>R$@f8`M~gn_`~N<VRbJw&r-+~5S9NaD!EFz25c_oERDZA!sr*}>H$;aK{{2$@?n3"
    "gD;C-osT4z)}!}lF!j9TB5y13L4eR)}Qih~}#DZ!)ra-Oq1^-VKO#Q}1Dn)C1aviPCkMPL2tyyyEZ+2_4#-Cy>_jP~W@{{sb^gxd"
)


def load_embedded_data() -> np.ndarray:
    """Decode and verify all embedded loss trajectories."""

    compressed = base64.b85decode(_DATA_B85.encode("ascii"))
    raw = zlib.decompress(compressed)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != _DATA_SHA256:
        raise RuntimeError("Embedded data checksum verification failed.")

    expected_bytes = int(np.prod(_DATA_SHAPE)) * _DATA_DTYPE.itemsize
    if len(raw) != expected_bytes:
        raise RuntimeError(
            f"Embedded data has {len(raw)} bytes; expected {expected_bytes}."
        )

    data = np.frombuffer(raw, dtype=_DATA_DTYPE).reshape(_DATA_SHAPE).copy()
    if not np.all(np.isfinite(data)):
        raise RuntimeError("Embedded data contains non-finite values.")
    return data


def exponential_moving_average(values: np.ndarray, alpha: float) -> np.ndarray:
    """Apply y[t] = alpha*x[t] + (1-alpha)*y[t-1]."""

    smoothed = np.empty_like(values, dtype=np.float64)
    smoothed[0] = values[0]
    for index in range(1, len(values)):
        smoothed[index] = (
            alpha * values[index] + (1.0 - alpha) * smoothed[index - 1]
        )
    return smoothed


def smooth_runs(runs: np.ndarray, alpha: float) -> np.ndarray:
    return np.stack(
        [exponential_moving_average(run, alpha) for run in runs]
    )


def summarize_runs(
    runs: np.ndarray,
    uncertainty: str,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(runs, axis=0)
    spread = np.std(runs, axis=0, ddof=1)
    if uncertainty == "sem":
        spread = spread / np.sqrt(runs.shape[0])
    return mean, spread


def configure_plot_style() -> None:
    """Use the same plotting settings as plot.py."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": ":",
            "lines.linewidth": 2.5,
            "figure.dpi": 120,
            "savefig.dpi": 600,
        }
    )


def plot_metric(
    data: np.ndarray,
    metric_index: int,
    output_path: Path,
    ylabel: str,
    ema_alpha: float,
    uncertainty: str,
    error_every: int,
) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.3), constrained_layout=True)

    for group in GROUPS:
        raw_runs = data[metric_index, GROUP_SLICES[group]]
        runs = smooth_runs(raw_runs, ema_alpha)
        mean, error = summarize_runs(runs, uncertainty)
        iterations = np.arange(1, len(mean) + 1)
        color = COLORS[group]
        marker = MARKERS[group]

        axis.plot(
            iterations,
            mean,
            color=color,
            marker=marker,
            markevery=error_every,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=group,
            zorder=3,
        )
        axis.errorbar(
            iterations,
            mean,
            yerr=error,
            fmt="none",
            ecolor=color,
            elinewidth=1.1,
            capsize=2.5,
            capthick=1.1,
            errorevery=error_every,
            alpha=0.9,
            zorder=2,
        )

    axis.set_xlabel("Iteration")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, 100)
    axis.legend(frameon=True, framealpha=0.95)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Plot embedded parameter-strategy mean losses with error bars."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "plots",
        help="Destination directory (default: <script directory>/plots).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA weight for the newest point; 1.0 disables smoothing (default: 0.2).",
    )
    parser.add_argument(
        "--uncertainty",
        choices=("sem", "std"),
        default="sem",
        help="Error-bar statistic: standard error (sem) or standard deviation (std).",
    )
    parser.add_argument(
        "--error-every",
        type=int,
        default=10,
        help="Draw a marker and error bar every N iterations (default: 10).",
    )
    args = parser.parse_args()

    if not 0.0 < args.ema_alpha <= 1.0:
        parser.error("--ema-alpha must be in the interval (0, 1].")
    if args.error_every < 1:
        parser.error("--error-every must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    configure_plot_style()
    data = load_embedded_data()
    # args.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = (
        "train_loss_fixed_parameter_Thm_1_parameter.pdf",
        "validation_loss_fixed_parameter_Thm_1_parameter.pdf",
    )
    plot_metric(
        data=data,
        metric_index=TRAIN_LOSS_INDEX,
        output_path=output_paths[0],
        ylabel="Lower-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )
    plot_metric(
        data=data,
        metric_index=VALIDATION_LOSS_INDEX,
        output_path=output_paths[1],
        ylabel="Upper-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )

    print("Plots generated successfully:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
