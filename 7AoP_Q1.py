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
    str(Path(tempfile.gettempdir()) / "r_comparison_matplotlib_cache"),
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


RANKS = (4, 8, 16)
RUNS_PER_RANK = 10
TRAIN_LOSS_INDEX = 0
VALIDATION_LOSS_INDEX = 1

COLORS = {
    4: "#0072B2",
    8: "#E69F00",
    16: "#D55E00",
}
MARKERS = {4: "o", 8: "s", 16: "^"}

_DATA_SHAPE = (3, 2, 10, 100)
_DATA_DTYPE = np.dtype("<f8")
_DATA_SHA256 = "3d3a3998b4e97ef1e7c746b3fc88fef7cd57a580e74bcea77b486726825397c6"
_DATA_B85 = (
    "c-n=1cX$=m_V!Sv_ueu{N16)K$sp22qzZT?Ls1kIY0@Es3W6d<qzNQL4ZZi42^~W3Ejg1&AS86s1IZ`r-D`gC{oMO}{K0<a<eVvc"
    "R(aRj>s?#Z-u&}3-u&O)597+d<=>2YN}F<TM*GisG-`FH7pKo_UMu`&w7TZYPscSj<e|T}&z6hEd0|=T?{)Je`~NdK`xZWVGg{9I"
    "zj!n1tGhjTGn#e6p40fjn)`1?qj{N2Z$|BEujn_UJMqPJ{66Y$`hDVhm%hKWWdt3UK754Em!&ryI|pCxr{C@P<~E(Db^bea-r#Z1"
    "==gG<82a7UJ@NT}60g$OCuke}J$tuJbUb6zZaPm!N&4IgnD&&$(Hn#5d#yU*d!3@9==!Oj-+eRM8@{|n_jmJ{P2W4egs!I*pC3Z!"
    "yL61Mrw`vx&t;8!7Du078*-C=ckio*be)Pt>3imtiS%6BZ|ASj^&0$moPPiLww3hveIKu<vC8|4XneA98U5XENb_kf4!S|dQ|8jV"
    ">em;=()r)cL8F%M#C-aE;{6~x{|`&?oS*bqK*w3}p4@GJVH~vP9^H3R2YPO+_m@ZM^CoS6qx01~F^#VKvpI*x^Choh{$Bn`=c{;;"
    "es2a;qWO1cHKO};4m6?jIniZy(b#YgozEE3g62;zVBr0&`<Udy9zPND{9->HJLf*VOk<;LbUkZcN&36<zVCPX-LT0dH_q*nBp2@7"
    "+H~L6w%R^AZh4EYZ||Lh=jyazDxLqkc#_Y5KB+)*{j(p9Mh*Y6Z$@`-Mk&&duY!uv@u-YKbliMd-Z!H;V>0xiTare<EA=3ap681b"
    "DRkZH{n8<4Av7=Uox!9ZTFy$bke5L(>3(*sqIqyj<iR`zJ;1zwc9QhM$$sD<eeYGzP&$5eZztXV$h5QcT%A@W()i=wkpC`k?V#%v"
    "TTk=q)VxOXVE4^I_u-sqNAhAE{}0WV{_FHG{H_hjp+0-SHTr$~>=^6pCb_XY<c3@qd1n><e$o_r4*h9o`n$g56Vg}h`|*%-^CHQ&"
    "?aYV1_DNp|z3TXwj!z%E0)77V5#(&ZG5UL>Xy`}R#iUn8<4t&u`FWSq^|RfF{dg~!^wr7#E6JPIcmU0tKIaIo`)MSecdSPDYx?rh"
    "`!R?8OxH8#WQCns*6~;ReWxLFFuyk4m%Viy&6|F@6}>knq#@~zJGN;I<S}y|ohLdf8g?O>^w$kB@LnSqVcu_kdkgllJ>;|gbIe1D"
    "<@El{Umw#vT7%l+{a>saPQTldJsNU6g7nkrTl+Nq-dZ%9#?{bEr{guem$K0`|JE@-(hDQ#1lcWjcnDqBDddNFUsQ+ek@k7s8T5Ic"
    "wq&Qwx;vn+h5v?~e%?9_`r(q_(6jlG-!R5ZhW{vDu?*?O^#Jl4*2Ca}qz~hD(i3xaSONNc#Jz0vd7UAzX+C?`gx`31lJw7bXF84g"
    "<TEtSR!-cXJ24OJX<}vAg?Vq$d^#<wkeoOda*;mRfvqu5-^S73%|>lV4(uf}?qXinlAUz_Y`2$wSLeP%pMN-m-iww{gTAMwRiXQ^"
    "vQMMucfOd0_xeL_{C#|mt#rQ;Uz2?6f2@IhdM_8fAMJ|`us4@OR?~6Ec643iLbWw?-aJF-e%<%)L(dkTfPDVkhV)tc^d#A9x5^pv"
    "r*5;FB;RhaK{#KZQ;TW57(({NZrYaKyZzB%x~|doEgXM%AM$#xHT*({!RKk5+75C!t>zZ`o?DIX+q(PU1YO5HLH5R&cmEF>+vFs>"
    "YlOT7|6M5r`g!da(gUq*9Om~3uB%u7jqHFnY6kgfGfPeSyWT4t_TjzLm~Ue+$-Pye!xB0!vyR@k-8wIQ&v@CI?75cr6xnIxdRy3`"
    "RyAN(D!1K%d8tP8s7+i$e%w7(Ee3k~7W{I#wtr*3Ph5oET}$V2Kgx-D>3s_G)g9M$Z?A#;%{hbT`ur{OYud=XKG@$l(jQ~mFZ8*7"
    "s~z-6(@7t+Dkl6$Z5@838$LH0X(dPxs#!%zZz}7AtK9q!`4QuB@LM!q%gmtX+YyjT*I5|w7`P{s?4P~NpX`>gK9uIcJ(P*(JK}>L"
    "?e~L!ToD{kzgv<?&tZfGKtB#=3OnYb_n<EggdJR-3H!V@lk`K|=}+@#uJPgdHiQCa?DNs}w0QxMblz3L2{iuYhk5+o5A!r90QxcC"
    "e<S_f_J`cf^ugXP4kmkN&&a$$_rELPD&Ctv?9UG0ao9Z{>5Xw982(^Q@KXBTi9mXP_Q7C!pL(#5<l8(IO7d#%4W;`w_WI#D*ZG#="
    "`2$YU{U6Vy-#LeIU&c{C!dKe$OyHXRfp=lg{edgOGf7X38=++HjoUu5)8>U>%*$Cnvdd;<D87H)kMzUx`RG2a+qi${vLE#KY#`ZZ"
    "`>g+Ocpe`;mvJ~0{^CX`y)Pp?6ZRt<^QB!0T?IMyho3kVK=xZdgXb_$`u@WE4xUBdzvCY*?jN}DxF5Z5>!c6zaa;WTTIkQfslH=)"
    "jsSWN>lWn3ye#JHbRg#QTIf%hr%bXV?#W<!-`1G`o=3cgvjGqOjj#0U{xmw4Kk0?j(IWq$#rGrqFgApd-!MjH$wxS=OGS!1tfpZ("
    "==f4tChWy>=;^X7<hQh1%Yh?PG_t3*76yAXN5^yAFzG(*L>qpvNS3p(7cSu*^N>lS(J~DBU>VS-S0>4;5ea#6PV(pnd$<>J>MZcb"
    "_W}Y5huBAbh(B&&-mHhdyO{4_!~yp+p{K7x5icbK!hSu^gdOs;VCVX{gxj4CCOxO!(xSM)YNO*lyknDH)B70khuv_j^{^>E(I;ti"
    "AI|qKa7%xS<X4-n!JkbuNuTXu?m@%{81<Gqy+3Oh=F9G=(fnv%IW(_EPxBu9kqLX-$3^_oO@kf}bm_V6;m%^<QZpF1&j8MBtC2mk"
    "COYsxeJs*D_kkS&94XE>O#@!vXuxhy)am)mU>))rjPIETOvE!<Kj7sl{ph}}5mOK+v<ZV>-B^+I&N5n(zcc5xr1>!u9lE}@aXJ0n"
    ">Aalu%3c#i{#whi9P`$C3dy(Cq9yccc}uci`hY3;UBi}$_o_va-rDbE!S_0bt)t&(%W@CzZ3<oA$XoHA&<~yDyu3?sxHa8{o^;d^"
    "m;Vx0lJp{^CHW7t_))@D_T3-Jf7p#y<fQM{%aNV_{?4z!2j8uyxIkZ&heqe<Da5lKH$z^FcZA)JcoYXaUK{@UqpzX&?at7B+QG3n"
    "j(b4*<fd$eACD_edgP2Rv={PO6!QQ3pRh+o#=_p;A9fRC9mM&o>ujgrFDMQ_5tASOzHE<J*xjA5ccTj@z`k98pKr7n_@ezh;DyH9"
    "uE7ppIYaj`@hr)`nZBFsy>aXr$){2CDR9i?KVZk_4~2c*|L%3j(H1(t^SBo5Me)Uizx48T>390!tt8i4Xac=|qw;phW3Qt0o~>rX"
    "2@g0;OAzj{C)|3BJj4jZW0MvGj|V)X^V!w6!tT}?N^+;?Y(m#FS0&*7HWa0Kw_1%NJL4Wt0v_(%fUc)iFGcZ+v%5QePwSZod-P-k"
    "&6hTH9PH1-*YK}p7C_JT^nkybyoH`q@A3uk*6V%b542`C_tE#>uSa@nPx@mX>}Pk<FXyipbgZ{X-4D6HjyR@Iv&r=P*u{kBtU$yg"
    "?!o7O04Egz&fWYu@*U+1I`B6|U@vlxCH-)hCr_u}?|ni3#oqPVKIn%5eaIe6e#1;UM0#OpS>Zpv7)0@hvvC&kRwMTizB13<ApOu6"
    "3?cuax1C4vhr1;xE8(wSY6Dkwn*w{3Sp6B~V+ZtL%@yFlUQb9rtgD;I&+DJKB)3}KK{Rjn>lVl}EuKlZ$Xt7p^vu{1a)^Fca}@Gi"
    "#hX*yV@|RWkM!ys2DyC-ywQ9R{od^`8+dme;&|gs1H|dG+aL}a+ZFNRclRKNH+ECL%1juCJXy;kqz}4zp5g}E7)bMNk7)v&Hz)_<"
    "=FCdSa}9h8ey3nZ;Qyv~==tpIUjb)a-cI-Hl=>FWS+)&uPOp5h3q`x&ch$dNfq1JE;t{PJ@XLog5l{QW{~3jMTmeq}9dci>1mfmq"
    ";bh;P9G6MX?4l2VSDI9X9r<}W{@(Wq^eC>}E#y@OkscX^N@1QSYk1CHHgL)5A(S^ThTq2Z)7nG-mbb(6{bwBPQ$ix_Vemw}=c>?O"
    "`$Piim$vOS{QsZfbK%chZlLkSL*y+-C6S-8DolS2J2v(g#2L6hC&x;P>x}|!>HGFq1z|U;bVdHB?(e|!+uw&>DRnCq`9R2}btM{l"
    "TXg{J_#(ui|6Gam!_15X&hmwlp4cmzz<(^5O!{FTE=~GjEL(*9RpDg9Sz5mo@*hUbE{Z>lL4A^tk640yRrPXoUwXmYl-JO26o(xi"
    "y%5)5d7Ivgd+Zjzx2+81T2BQoX>bkxSi3>**$hg=duVYDIA>-t(j(*GF4(uqjqrXJ+=QGRShyYWXGz4L*B1akwJg4tu2-)ZaNU5>"
    "usdH~gMaN%gyh?f871P}WSnnLG0g9%&mb3bO3;0q#%Rc8w-4YCdNcrjn=ps;Qs4L){NL$aB0n^O?32B@kidxz@cgg3(S2z7#~`nH"
    "bvWd-*k;7@4=y2Z_HB=7e7`vIknhZcKfBQhc4TB>dOmI3Mc|mZ4RF6_<}QRC8io8*skxZX@H*#_&nOH#KPdtBw+QslYS-ly{eAfF"
    "hv9eTA|FzI6rQK?OPW`=UjpfucA_(I=YWeh@%;5pA&)Yb^1#|(TYv+ajH2(mk)06Nuf0feZ}0ex{EvBj1o9gnU%)(U-Hbd~zM;q~"
    "w?p2<csT@lkIXBG@8(}bUZQY3&4;sWF!GQMFHVF1nuUDW+qGeb3tfia7(EAgKlh^zB7a4CVHI>pAKa2oS%J4oliat`k>BlZlYZF!"
    "9EwAn_YsF^7aioUra83kVSa2N{-}yL;h*zCPjnadq&RS~SwVY=e3Sz__P&8QQ3t-(8#u7bZEe_>4(@6AeGT^g6BoFuo(_HgPy_yM"
    "p#c~4cY!}U>ZC_z7Z>=crwx4csR?=VcU)l)ktb|!!vB5@Ty1s7=gv0<@bEAl`P6<I`E#d<NpXkKP$L|vEw7Gsn3ZkmdsaYI=;NfS"
    "u%GQ>fqM@Ph8^BnkZ_Y*dob2bc9o*IQQK5-0rH$rkOxlM2)w`T$~pRbdOYl1kIRt%)gM4kf-k}Exx>kxXpagZzCO7J*BknR?65X7"
    "4LGRRCCJmwUXa(2NaVw6G)JDe%7l3M%ks!03@c4?uNO-|d@<>LdXM(1?#KuIihP4!ya8}c-%G%aXC}bT?YIT~Yd8k^iBtQLS80_-"
    "pPRK>QU2EW?grLd8uo*oTDu5w_VYgQn{kbiXWJDG{8+%6fWIdouQ1_&h_9Pq-lyJx-Ac=fJcfB4{`ufM<j>EHhukCyyAxglxLUKJ"
    "A1%VMJ~(JC)&<XgPWZ|R7*GDgtXrJ)!~OFG>4#NzF~#ThooK`#sV`_9L{BS2{=*FKM0m?wQjf0hmW~0={JdcT>|qb`8%DEq(j#Yh"
    "AFR_fuTAICKKYX7-ROD~eqmxP>`aNPSXUfg3;6m*IO5e`{pq^enuW+GZz+uRsE;CGx1&b@r=%2J3p?6_;%;l}df1PAA3nzW^52O("
    "P8#f2|3VZOYrEms?SbziFZEu!4{=aK=zGmIYk&vl!QPE)iSJh#0=-E~MSPT;LGxl=oK2rwOKXx{x6A%c$L^k13t{i}Qyiz)*XjPO"
    "f@ko&L++D&8*^6x7sL&~e2kb!d2h#9gZrs>X&v(W)qrciXn?%i(T9k`?tVq~*4bJUamVooSO>h{9r;mx0^;H_zeiy`X+i?xh+osO"
    "?lu#7uojo-{CfNl{C;v(#6P$GL-tI&fVf;gy94=}QI(L#tPFXzs#K)sweLKbf_%qqvbWZho|wPZPx1cVd!K&qte=B*wviVpKkRn9"
    "LGilwZcD`fc{d~hPxe86?@Bu2^Rr)Ly`{`r*x~js;7@-4hWw<Jigi>ge-Fe{d69>Gv(ioSs5!MU+I29xwZ!K&F&b6GSP7%4i|>^d"
    "pBEEjQH)k@j9P}sBj*z5%Py|-3ZHAx8_jts@Lsg=YneiSUyI{MV!R>7Y~ntiiQnB4*GUq;OBcuD?}m6@Cs}-cT6`|PXFeA55h>1}"
    "D&LoTK>Uu6#rmpyTYT?@7{z@U;`{bfaV*xGj8HKOd}}-r^AIO~FY+eZ4)OQP;&X8xTb$RuD()vr-1j!|xp)pUR-9*#I8PSwULMQW"
    "GcJkq#E9#N_u(dp>xucap2_#i^J9p4a1xaH70+jh_h*ae*M&ToJTL3z_oHb-uEg`$;{2vqkI=+<?I-ekN51d>-nTC1Ps=LgGFrY5"
    "ce^+~Ca%x#Pn=(0F1{!9!W4R;za!>Rj80|oxu1~t3SulPzE?ziFRzgMf`%;D;{JI43?cVA&#M@1@wp}BPmd7KohHwpE9T$2A?uIG"
    "pK5lB7{&9Nk7RkYkBej0BOw>McrScx3B3Vcd$Y1Ee~&-cg&dmVe5QCFo#ix4T<<M$ej%^gEBX7zb#YyxKRS=Xp6G|g^Rpgte&x}B"
    "mrp~ye|@(&ub59u$fLbe9B&oJ;yLV8`MLD#vb;Miw;|$lA$R(I@mvYA{_*?d>#^R7JoNv`rMN#^;3D&m_&wXjC$fAyo8;^9^IJks"
    "+*M)}$C|i~#quP+ud`eVxzYLhVq|~8qrs!_8<x-yOX!EFN72Okh-L|Y;pP*1R!E#r$dM`L+s-K;yTYEao#Oj-h5kD!vi)NF>xg+W"
    "T=Cq(zFNW_@%f!Z*$%Q@(plcmi_h=L??o5qaUIzoxIDkY{#kq-A%8}k>_^y6y1T@&&<iJ4)(h5emS;oAy(#qAV!61d$hFWLP3WO<"
    "Q+^)yL)Hsf4@{ppkI*9#KWYx!-2<{6<?HEuUo2Nb?ksU0SIoD~cAWiFqCBsbus^Qw(+2y!SXo{XU;f{D%w+lReC)=^_LTpwi+Qtz"
    "oY`Atf8_9e3OO^`F0K&gT`m4zSN0#K@F$iSooeFqw~c?Tm%NEU#OEc2+!Yq`E^w5QM~ts!Kj#Q}chZ#SWxRJ^v8S1`eP+L^iTA1t"
    "zu}${pEDj1_Sz6~@5IUTr3t@k3VfoU6n__=>*BukBjWsNvi$ISX1!$mVhTNA`{2Bm<%r)4+gCGDd0%X|gx;BK2M&tA-;?b->y<0~"
    "l*2ee=%@2a{+*}KjN@YD-?5)H1wOG7WVvU2;|jdMc8ag#?2zTqX1|*#<1$T*jE@b07hUl_OrihIRoNdKuVg>Lc$)FJ=TA~)dtsec"
    "a6Hdrh&(_1ynMfUEm<FQF}lKkn2fK4ei-G1KC)hv5Z@EHN@u($p3h|b!*Rkx+1?uWmG~e=(dXMTuF@|n_MYb_NwM?c@^visA8fzx"
    "%l1@zB;#dMoJTt#`)Af~SKxG0_!;w+9LMne@cXcK%l@68ljAfiTJ|^QHW?ootcU#f&Emc;DEN=<v`^N5?SYKXEpgv$SB*Qeyz0XK"
    "Xotl4g}q_?#{Px<1KT^c=eBrHTDlw`@qK7F<#}a$r?H+tlkp?RefmyCuf)9D{65$p$H?+vUzhEXo~p!;yJQ^0c*{`Ya*Oemh(Gii"
    "vj5P8|Io_G{=+CP=1<@#Er%i53!UxQ3pswad48j09L9K5l}FYGwpWZ#8HafB-!nOm=J=EG49~wM&TFyX5&GktlHU*CuY1B@@)O?g"
    "gdUmVcNXi5uz!|#Kemty>#%|gI6iqH<4iA3j#uoQkRzS*8(c56SN2!>BRS5}#NW+Tvi~qR&S(66PL88JITmtb3BRFllkKf3?6J#s"
    "@3E}cj?imoi##v<yxwy%{^sAazkDXgmz@9M=Wtib*Z1(?Ryp6HMano{6aGaDQRIbj6vrXLpJ>8=*qpag^uj449~;7dxFY^AIe#VW"
    "HpgMsbH$Fml<m9Ad99~Py!f{qFLU0@y(GtZY_}MvFplB*;`4Iez|$kf<wxaw0lz0hT+fb^{k_iLzc1Sz#$Sf86Xr`f-(m^-t8JC#"
    "&dXzP92q9-Eyu&o0oi_QTa-N1Ls_5Lf7@AQKVS*mW(oW4`CD;c)^>T`cwQL?oBNbJig->ZQ<iu3PmFgse$aNxaR%czjw{WbG7jLl"
    "#lvqQa=huVyzQ0kCCh=mL&oduXBk&9o^h1;oc)NGzhXV9D&B8JF|R@&bjDen*A?-(A?BI$ue$ID{C?Q3F&<%i#r}cq1>0G+t1PFS"
    "|M2p&@v<FeJJ0XSWk1FKOT@ievW(LVj+?G2`I-dT&++>=g<Kh&4`#V|Ec+SncQLa4vyRBu*DlI&7~?yO^=7N=ci0|qzJb5baT3eD"
    "y-|r94#{~WSHxY;c6t7_eM&xTtvK%qS)LfzdHEF~w;Jp1HCbOhzxh(OHzwx=_<6+d>?j$pvVCQ{!nl;5$6&vHOO87|J0a%7S|$4r"
    "ejd(;n~TM$tbZ6nFU%V9x(DkA<13?-&;t>FSRy{RbI7BI57OlLiQ_4rC)Rt`BQKAYEZaTry}XqDg(<FQ-IMdW?5B8Lh4TR#>zjz{"
    "^_wy-bq*=`ELP61T8T>DB~q4u&R?^<aXyCat^H8OVP5`6iCg(|&Np*@*c7<LxGnn)9^K;#{!fzE1q>w~_3$>^X_mjua$dr_PKKNZ"
    "bwXsjVqTH+9L700?ql3+h<u36d9<gpK5?Al-jm}uj^jB$z;fpK?L;}QWc%agkMGNP#Mq$ZhdJI8`k`_DisN;GuUz3j?27WbAnS+4"
    "@p(>p{g3g2A>>+nDd(5?{ji*Sf4?ouy`3QA3&xXLnjFWoU*x#c5dOt|Ex$)^J>!`iw|F=_RL1x0kDYY+J{e!Jo%V3}VHx*(`D}6j"
    "7Te21@_S+&#_=`VHP&<16Sgm&-SjEzgi0RUTMvp>)@fqpc#z)*=hHdP=Ia`}<>zMoVSVQHH!pw4@_0*GS5R;^<1Nlt^8I=F;}`OK"
    "x7Ug9v!34-=V82cM~>6&Ei#Vccv6d2@<gg$%@_AU^6rfFQC*@H+n3fki~}t&&fbspj?UMCr`#K;U;Obp>a-f}qtQP7J;mu($_J=t"
    "EOMLHMU1U|QI}GoGwK_UoFW{i=edXYBxgU$13T4%2;Z3X!b!i3Du@sCZ)TERaT*n;_ioo*hxrRXL!+J40Cil2Lg~5f<Bh<VJPtmk"
    ")vFHbhkT#o`yB#N$8;_SzPDip>M;W15Xb!VBkHcF{s-@K&RW!;Y#ac7K-KS1cR$%hUE%rMWdGdaHu)iI;XTO1n5n39J6VqYt}ky+"
    "@6&9!2=y5CQQv3G+_M+_sMV;WY?Cbw_1uk6A2Itd>elLl-(zlihUZ=O67^5nH-ZmYsVVA|k5#AhSs`ah{*6!Xgo6iA0rmD%bD$1p"
    "VKeZyo}5Qr$nepyC%<lkzi9jp?DC}|;KghohB~Dwqrnr|(f~ZOomWxk(&Qp|`1yal0DeXNt>E?6K>g>#)~M%R3BI}8cpvJ`!cm88"
    "P1p)KEqj*ytFgNW>fiD#*o8W@38e3Or6#D8I`cj1hZe5}f1u6^)NLO<iTb^Vr%`8jqXX`{;P0pt>pB>I{=1S`->4giy3^kl(0jLr"
    "4p;+z<LAOoo8UwJd<1$hIEF^I;y$bghQ^TmJ2~P}=evFa)sxr}sC&2WE-HdLtSE{@jG_@F|K{C%;2k!-Kz_y;uoZr0%TeUHR+a{z"
    "<YqMe-hF;KD}LXV;thM=l63INGEr|cW(C<-XVF^JJ+|sf_{_YO;~Dw_96|kC{tcx6MnX^2>uf&-{&MS`n4e`uQTN?(C+VHN>;ddX"
    "(<Ah`UHKr{Tchp~iVvJuf)`OX9=x@k<$(9T$_9IM@;C4sD{V)8($s3e&4<QdJ$Y?&*uf7VFZQGi@O0~R68vxr_HXA})HD6O`abx^"
    "*I@tWAJ~dI-6uHjl1r#Fs?ZF5E;8qlJ+<_?A>h~NM*U#(hmTQ@d=vcq#w&rhTTO+2{x%fPJ*Wxl{c`;T-pSmdkeefyp&vgT1YafF"
    "AgsFwwkNqZ$|XR)CoV_*zu!0TD`nb)zn0YpUfdtkQO}*V4cP~)WLflqI57}>%I4YWIrP?h$j<9OenfR0`rA98Pt|V09`+qhe##l~"
    "F^%@bjrjYAqrn&MKNh^^Zgub;oBs`b(BT8<(Vb}UJG;eDy_Yq9J>f-j;K%enjavC(H)>77dq}Uw@jmq7#W?V4n=gbOR05AsU-&ud"
    "7tj1kqdhY_>5ctlCFzYG)s5m4XGjmiU)ERo!L#~vGwFq!-i7qRDBlBltxvB&55DY<J|`Q*nEf*9^#*qWFSq1Q^dV^SJL2ws`7%+b"
    "nn2&TuNKGiPkM^_+~F9V_ufrL-L9z9uD1#2JAaP+ixVNnb92$>W5Rd%d#+(Pk5dHw2g)S?7Y_ZC^u$hw|8hEiiuZHoPsFo*3z7aB"
    "ITG;x@74ou-p~N^T(Um$yEk!c=Sw1acXn-ozQuhH+?z6T73zgMqhG^}QK--SZSy(7X92IWKI%(Na}M<2MRE9#p+!)K|9T|psnc^F"
    "=Cf*j^v9@%(X8|mayzf!M${Sa+6KM2b_snkYSBC!MK33VPkf&EUham?n5Un1Lf^|J!Eem%hPuh_3tZGO*9Y&SZ87-UM)kp~odo;j"
    "9;gSqRdx77@S76QkHzVP{xEm`0M7C+1l~fM-8THfaPVIXoezRPE<yOsdg~gkcRORAuSLCT7qV~89rP!$$`k}&ZN)I)@ej8H7gTrw"
    "z3|(%5B2`_!MFBp2hYL3JL5URS9*{EKhi7=xawAO@IiuGfQQ~G0{Zf0dH9nbTT*`4NUBb_%FN~hS6%kOAH-cDJmoyvN_ye;&q=&0"
    "yIlgUyIN<<5WX_2)}v$RRU`07whtsfVZ^RRooixm+-LY_@T-+i5x&#|vxCnQcn5s5*T)Wk$M$SH`ht{zUAvnAyI1ol$%7MGjO?RX"
    "HkELM{#gs;Gq#-v9{9WdcGzpdXDhT4JgUNXz}Fq~A^6FK>VjW5u~r)FeHQrf)7yb-Dm~1A+;xDz>N^v7c-Ls?Z<d+h`Me$uo<vpf"
    ">&@anksg{|KZCr#_a1QIle?(T?TWrYRvXBV^<69UqsR&Vx;7?vCV172fG5|yqWjm|9w2!&N-iTgbw=WIYh`u%zSa0o_{|Ylzz6yH"
    "2k1xg7VuK{7DK%9P6+xx%<T$3#*F#+-roI$i``f8;43Wp5p}l>6Vb0@Y<JZ42UJ9#jH)N$|KICG`3bXj9;)|oKZyiCYiS(4KQ}BB"
    "{N^$l+u`?50>4cgg1Ywt-!Fkb4gfyQ5chc|b~F6Jk_3D|7y3vfg&{8bWCHl!6FX8I=zLxgb<`&^(I@8Z&f)*&pSz7ZfJYe61LLmV"
    "&<`7WVJta{J`Ud<LmaX)Cyn}nP{bX%(xD%9N6_fb9*um~`%P(G#Qps#^x_ML;tlh{0g5-=3)^x0<Oa>JGk7uKE9+`28r{YDNpJK%"
    "U&3#cTMC|H7v!a^EkjU0pR|bVy<79yBk*DqfX6p1Li|v=A$WW5RtH{i%A>yi+e*MWFQ34!Jm~Wj{Cx0T?b36>zia<J>cXF0Ks|j}"
    "ufyPN>fqNlX%AlGzI?l(CuL~f^++H5T&XP3@1`HYe@*-vys80r2p5{$9z!mxE&z{l{!Q55oq2$_*1wCo`=Rr|clmph=;QSh?A_-&"
    "`dQSg1Ag`Cg1|Y0>xF~AHy=EZ#JrHZ{GS15?QM>DEIL;3(qAB7G%*7G7Rpyyi+(`M!P7sr8FJr$66`^i_K1_$CID~#Q3iNvd0XJx"
    "a-u$F3V2Fdby5G2G#Yj8<)VS>5|h9a{N*I!L~~*=<>~BZOY!_E%fTmXIurVRtIAUF&}{ghi^0g3ojD8tpMMqX;<q{Ak2`+~zg6WN"
    "`W79(j=oe+Is-@D*p2w+_qTBWWw(m_4f?}b2Q9?s-&6#?niU4UxY!T8+e#~Fbp2OAZ;BqFIK(Q}pKz4!*Pr4G_uV7li=1vre#CfQ"
    "5qy!SQQ+n7ZVBG{&X%MX`W_v1TJ`#qe(1HnqPW2PBR6o@*PBTH&2QgJN53k>SyuD&@P|z{qOVDJ*kSj|V8kPru7bBWb0+#6tgcJ;"
    "R1d8U-oe?;h!^akIM0pW;E(#`#rrIih<+~<iokv+bOXN0$cy-A^Cs{b{{8{Hq{Z(7SC#n|@#pLd;A;&7Z_XK92y)l#HwS*J5bWHV"
    "jfh{8F6=?yJHZ3pR1fiZwk`DDtdSQe?$!rxg#XHd@7XyvK(6lPrFpO`#-mSB?mvLve%k^$o&G88!_)7;Gy3y8@V;($gxzgg0P=C6"
    "9B_Tp3b3yM6~KF4T>-qAaskVLUnavYt(pu!GS448u*y36RFt+4!r!7_jrk-L_IVWY_Rd=H+05l8{CUy05l8*L0(q9=D^Mr6YXGkI"
    "Rwc|!&H><s4LS;(_e%`)DfwscTT)&E5Bv}WKHKWz;7?C42b>um4g4{DGV(H^f59&0_Tl{>^QU=p$2(X4oxd`B1Viqlz_Ybt!oZW7"
    "n*;I0wgcd$7d=6vbEX{P^k!KRZ`YZEb;8x9kk4v27;-oA32^&Io#BVCPbK_rHYg6iJLLiRJ=>6Pc0PCq@m2a@;FKR?!B^Qi7<Qp3"
    "aIJZv0OAurymzZf67XjBCg7j;Lq9zI_+j)H7<UryukIr}_r8&^gJZ{wJg8en&Rg4}E}3~Nygn)P*{gf?>gQj}buyffHx*usA?96+"
    "mGko6x<2O{Gvqv{qwo`W{n)EFdn~UPx_jjHQ*T{>>j9GFyf4@3a6KpIpEadE$E&aBdIw%#5qVdW*BSXfSITv!e81*a`TmS8a$Z^6"
    "EZ3d!^O=|BbzralI8m;f;m@tB@;athUw=iOXI>ZKIvm&icYg-!8v0)8r^9gEaRvDm_vtizo;e8qB=}wUnY}NN-)%aF^h9s|g5=72"
    "ycswvYdp!9`|U%-A*IeB4yibd@>@=L7x)wX0?D6!?K$c}KR*k+rMDuyWHmd4cz)4X@abOngdYeljCG5Vf1ux1kE_7fb-O@cPK<(I"
    ">0bzTxZe)oqifv}CokBFJXF%B;B|fnUZys>0p?|PZTO2T2{;Zb2Dy*_HUfPjzC)jhl*h0~A8$s#o))!$Gj`U4-JO0F{$wrsP#8I9"
    "BM<!aJmQF1wJH8Kri=jJw|H&%y+b`9KYp`-L)tcY4*qmk@TxnZ|AI5K2kxt7UErVd1%RJw)S>fOn_psmVe%!|$<@Q)XA_=+|C{hB"
    "_}o>VAs+tu1@gUu_ihy}f_(K)Pr<kN^5q)jCuW0xJFzI{v0WnRf&142;P+ukz?%<CAl{fd59^1QFCsqq>l*5ER_%lx|6xCH@U0&S"
    "59(Qm;CVYWqPWtIJP-d<#Si+q9sLZf#c877U~%w@H_rin+Flmx{WFShM&6+(UC)fDzZyJb^z|@Dq<~Kwoo6Zfv4kPNxMsRoKNozb"
    "zg)!M@Af0TFo$MAzN#nilr|_D{v)72^rOVP$X{)My>$}b2fmu$1pQ7vDTp}z?bpa#?eC5LS)EFwZ*M8|?R6$RfIf`AhIqqQ19{vM"
    "T@arau7dT(m}i;j2Q?b_sqS&)cc$$DpLTzJ;LNR~&?l>TL*#P{AkSjzKLJl&t_FJ?oCMrH>pS3;>%dd?AEN)z-gNNY$8A9T{84$x"
    "U#@qdFN=bKt4_QN-2X>DjCIQ(kMpuHa9m;Vdh`|J@VSBfnYH%}<e~O9^fee&75#?#?8N(TXhGjLcffP@xd}a-&=l(-lYhjz#OA43"
    "uj%p%JoLwnk&jw&5q)W|mWEy|jfP*f6W3v#VzcNgTn729Jn_JPuTWp;2A#W&ylD~Sl?#4?^_U~y!0z_Eh50LZAJ287HR7RJx3Ny%"
    "F8%`YPg|h>7ba33%KE-F>`VUuitFsu7Le;-&>zM9VKm9FGpst{Qu9$y<f(?nA)mbFdMf$~ei;Nimx^_gR|}UTj=v54UKfCTM~5C*"
    "*Z9#N{OLo3&Z1AqLtMA&mx#Z{+#?)kepwrMcIHou|Ls5A)#c$gN-u@KXxkledj1QD!{c(`d+7skT;VIiBi8rX5T93Ghq(Om*F}-9"
    "%|rU3f1iPVSphZCk95ZjjFsvlK3{ep{ZPWDpifY@b--h-G7yjFizENxtZsw+N|TG|cU?XK`J=cIh|`nfV4rhbM}NkXbJ4eEO$g$-"
    "7sueI3tYzf!KFf?Kkr{ue`M#vyx3jeJA-~(m*9_IB_OW-U>Nc%1rK7Lv)_wA-g9>%aIA}UlKMf&D}8(u{pFS;&eOZbU_EeO1oWZt"
    "9`uW~?jSGoau?!&!X2@0muDIBJ(V8qhrX^vf2%LhcUt=h`DN?GSlsv5<B{*0@&xN~dBM+jUgicqtr&uJ(&ZD-Ph?~r$Wgt?=!>&;"
    "DfDB~XYgxNP$%GImVzFY%Z51c$8UiDFQ0`S>CqGMPuvIR(C^?5@(zA?S0aAF^BSe%fTOx*N1T(p2k`Bjc=(5nw}F=p;3(_%a_HO8"
    "<%kE|&w*dB#*kmI2iGG#*Yh7noOiM$?EX{K^*B}EgMNQl6Zw+VFOV1hPcXe-H^(yg$M)z??3`}^`#5G0^2wD(a$Pa=YPk-J>rIrp"
    "6ODNx3V+IEo?&&R9)<Z-73KOZSMU#YF*5FV1drVn_41CWTX9AGqb7K~w&3Ns%wy-cfa_0lDs_?qCtF;<ktp+k8259%AJ-Q!AC>Vq"
    "*G+hJTO4m(RO-AK2QxoP)VbKFWxf#CEi>QP5Osnk*Iize>$p6=tEd;&#q(%fmwQ*KpSiEpO@_*K>|A%kJS497V4fA%iyB;?d|0W!"
    "h>`2IxSopZeZ2bWM43O&@e$We*$3r%G_EIdLuLK}^ZfYzd3E;uyh<Gw*Xgi)Scl|#8m{MZmGhV@<vK>MK9Ti=^`V}u2bNO5sB?Xm"
    ";1}vbFLc4D(u95hxBWAUdS6TEfiCpG6mqYN`d^kmt^;Bo3D?7M9LDm&b?IEU#qZ1HxSH#op2+oJdbk|_+RUdCBlG?YQRk<p%X}!7"
    "Kh_I=4_sHx&%yB@zfa~va~&2x57$+4J+r-6uB&3+5Z6g@9U#{|GVhV=>on%my_V&{-mhFQMxF;#;njNl<a-J)O1v+YORwH(kHYtU"
    "tnfq9<hnGD|GD1;*Kx7ly4PeL3G;q<9y!mzeF03BAHaHWTH&kmJg%4LhwF~H{?w~uSNP|a!UysE4EHq={=#8?i0}`3S(zVV2tS|+"
    "ehBjlnIE7hEBO%4J3mo)om{thQ>imelIwiA-jVGu+cD<5a-Add*qFa+D*QqAC#GU|xDM8gm+h+e-X6>K^K8Fa-v596LFPl)+Z4Xw"
    "Q#sGW_1(_7|IQ!O#B*6uN*z4cWwM@Zk@*>H=h^>i!j9X_-%6J4J^M?qPB=#9A+cTIIzeyVZz^?)oDWp^gWOMr^LN^IrB3>loCjn6"
    "8|T}&o|OA}cy-q)a{V*&2l#z?cDA<hUvZY9^fl3H$-EH8Rc3XWC*pD)Jo80_|FC%!dSePbaz#Fs`vYn0Uj)y}5TnNarGWf7^ENo|"
    "%6jPXvA|=@N9Xfe;_vKd%>3ef;(KhrG$FsbxUR`Kj`NGmBYv*%{@72&D)ydn&mFn%0`vA+-t3n$4q~1=^Qw#sO8zrm_IDnS<cZAp"
    "VO}Kj9QbqQM{r)&x+M4Y@puR+a{a3vs`P`|uJGg8Z||4=68jll(G%t;d${?jd|!+=8K1ftGOl#jDZCoi-(7NE*RvnYzY{zM$5!&>"
    "jGL5rgFk0p2<wHd)W0*&(DNe#Ul{^lG4GIZ7WZ}FzFD3=l#$~O#$8@N6y|?uGXD2?R~a%N##=9VDbK6te;L0g$nVqP{#xwM#C1Jh"
    "=M#k=eOZ2g9)5VJ^lf2W=qh>tBT7G%M7b{)<39FRj03!W5=k=t(pA3MeYua4=NDLi1m8)^F0bcspCRX<!bgdf`vJK}<@E^WNpbvT"
    "GcJmh*Eg8=Whi_AZ+(UNQTt>b6|YNp{Q-8%^T6vn?m-!6G7p*iCVBitpMt9%$b2&PA5JYf?qGjnD}T3@_`_g+x`;#UO0xf8++_*<"
    "Fw4tvImhEV^GSps(S<!Xh2OA+y|;wCSy>gk&hJIckHc~FV|m^f_n1llofl!did|-&-ZcdeWXQZkkDq)0zxP$+xYOgQURCl0Ph>mp"
    ";gb{@$8(&|>x(RZ+6|=-(oMOqg)RCS*~}L_CgW3&C(JzM6LQ?keS~!G!=c8@oM+(tNTl3%!;Db)Gtmm4GEwI9GXCVagZZW&e&+Z4"
    "Otvp>v~u5R^1OI_{3i;(=$?!-J>0Iu8=A@|tS$Q$RZqP5g!QPVtS?4&`PdRTjCqN=7`e|D_e(KZ&nnCLF!oQDICd)j_dJ<5zAfWE"
    "?oY($bNHD1d$FDtmGcIMz-eq3z4dX%8@c8Dna7)CI~k$yz@z13Ly0%N_oL(i+;CZbJ$wFI_Fv2+=D6LAll_+G--Mmv{yu7)Whwpc"
    "Jb$I+35;Z=AKP6;?s(mLm)!4-*ZFxqynZ9E<<B{u^!k$Am-7#Ne(w9_GM_YA;X^%F@;rx>{t!`eKFe0(4lhro<g;AyckcgV3Z6Rm"
    "k+W;a`CT=SW%|i^ER*ARfwyet>2iLzu$<Rrexk=yd@b{>9r2zyZ^C|_`GFqKI8EuJ@=~^6Uf<0Kxvw1i1Gd*J2aHEN-26!P*Nnrw"
    "IFb8eD)km#-$5b&8ux)Yqx4;QuH<K<ls+#P<@4J<IX}U99FJEXE8`rNTfQIrs?zWBk&=ISp!8Sbc;%ox4>rd$S7d&>7hk2w`E*0+"
    "o5FcR&I5S7$_Uv%d3^!z%YC*irLIR;>>Kx0;=glVg!}L6N`E-753drRv;Xk=kE%S}Dzg1&KCaFq_dDS@y__7cbDVB-9IfChuP+n#"
    "-Q>JjvMm1|PIw{5ab7-^{XWO-irzSC{FW@^E$;uw_<-}29uDTdaGY<uCBG-exn94^WF^kMrr>0yACbX*5>Cqf-86xtJb!mw?ic6D"
    "M}&Nz+TU{DL5}0Oe=y^B&JS>&itU9izR&p!=IwJm6URArnBq5+Wt_>l*puH#<-VUO>s9w;d(ZJ5_rG)*@9mf4J+_P7H-P=H*B6!T"
    "h3Yrdd{!M9Z`q39@NiTWIsRb0!+PPAmE&_>2jM<*8v767FDwyP*kWXPWW8X2;Pq2Tm*WqP<Jk`Kx(D}9WIO5gS$Zzpc`tv+_&q|w"
    "6#`Fje-iFP!1-v-4|?@>kL9?;TTfv<zpwPC;q^}iH?VwipHkKb`-mKW^Y59L&+DmXhSJCJjEwg^{`fgLpW*TTBjoix{gS+n==o*N"
    "Qyf<EP4P<Jm-{X&>zih%(yxv2f%D(<%dCft7rphTOY-|+yz1;%`u$|ceSf_^pQs!8D+KG=d*e`_n|~(wQhDxUJn=T_PpY3oUDnz%"
    "_}#Qe^nG(jEb2~jrJ+7+R6Eo&X4#H9;PO2&t}2S}&DxCLor^|&eBmFkpLETZq%Zd5<){lzjv#$;Mzo;O%#{swT`N)7rH8df9ewW>"
    "sPAfDiu6f8`wRa5Q)@bQ4$h&`jUEpk%6b#_^=SZ&uCb2tt9s9l*zctG2Gr#@*$&=(ot)UmVMh@5ak$qS^<rxelAh?}{jncWqjTVQ"
    "MgK|hq~1jIxy)G>brKaDWyACCLH&kbDb(|q`vm;Yb*RgAN_`02RAU_eUb!Ob2)mpHzbdo_>g!$}K)rp3rKltNbS&g^=JFKq83$pX"
    "vwXYpoEy%N9N3wCQ8!&F9DMrwZ=c4#8tqYcGW0P1-fJvf-`F_gB<lL|;`(<+fp=f>2h?dVD2#o1f_s3cuv+k-e%*um$_1$3u>Qop"
    "B=+<ysIRUcgng%%Vm~E4OE~r^KG6&F<Np=*>)SpX^_C%1u`kcuk>E9L%!c~;T6Iuo@~|QHYs{a9I-s-Nz(Xrt7<{#yjljQI^35L9"
    "HI4^w;;U1zgH!gvZXA6j@Cx?-&?_%Tz11d+ddGfb|DC2`<PV$;ttpN$rcZ-k_;NUn_TsEG8Yc$e`|}onPqJnk^eI`y8(F|JF@_h#"
    "exFlEp>Av8F_K5;W>3iFm`jkyF5$pUJ7&Ru)NV${&fT1JZ0x&@{d1$1lUy62z2Qgm4<h+AckRO%H6Hti_DR5Tq5a?~7VCm~ucx2U"
    "vAOmUjaJbp$Zs>Or<m=&z<h1}9(9kMD)RbO5jo$calc<)?_<A^MfO{avpF8{aQ#Ci-h3_h+wj&Uo+*9D?<sw!7$>I5>mQa<C(ije"
    "?$>Gx`<=jXN}Aj!$g8J+sPw;zQ~Kl_Rr=bWl>2~j-iYhzxev0}uPai%U$5V&BOi0WP{Wb?Bzw;vt<*cDEB#}5{r;#Df8J5*gupw$"
    "4_=gUbpZUst$~p5)ZEaU7jtoJVjmsr!w;~pXU<gcsb-5j(L#-ImXXIFyxo4m;H#hf6z9)Ah4e^Y_yXsh{VV#XtiMTmqt6*YqZ3<&"
    "{D(fRD#@Q7Sq9^h5cGA(8iD-~=RbyjI5Qo6Qu@D#eL}0hg*xEkX*9a+8<PAwtA2yN>^ccPXzTjGBg10o*lHI=a%m)u#(Z|0j{3(o"
    "*Dx;m7W<vg`xSit*kI(j;xlnSF$d^6dTc1pcPEte+P>$*KG-q-@H?@YbY0gMO4l)?gHfk{*9Siz;SYOv7yD-GG5)B3yq5{y^uu8E"
    "F?sBVI?IRtsQ11bO7dsK`=E~x0#TnEjeU%b``FL=O*YB^O2K(zeDGh<e)yg^uN~!wdQD$uJo+32qF&hNNB;32v~&skD*NCkVgsNz"
    "4@0q^o=@!S9Tfnc*&`o#nfC%=umASJUibnYiSvblcj*Uw7oUkb_n2VpYkM~s`}jTZr}$Zq3M8ClJ;MIfc6{J-@%&-pci1oIL1@Ci"
    "=i#5<VgKbvKG=hJKk&=q0^!G^Leu`uAN(iEMSXiojBW|k4_d50NrCVyW!?raq(nd9v3gS>=aEZre0?kSY0GjN=b5ts{HkC6!dUBL"
    ";H#nAC_Xn!$6+70dehLyq2vdIYwR<>pzgQP74Xvs>%a#uUjyfCTm^fdN38Qr1MV~~Uch_*WI1q9g94DR&QCC}(bFM6!-j&Vb_wyY"
    "^@rdS{*WK<E#w*S(H9A*BY%3}IO@wkz<j3q123iqf=8Kx{jseyKlE)$3q_rOdM5G=siE+9DVeD6Ps^nHcT;@nsPhg49!m8?pN=#?"
    "?6Z{;h`*=fJ?iP;sq^PnT7cM}Dgd~L|DGBQ-e7tF>W9;PsP|5r3LKK|FU}K+dgFBe=i)rj!*tY#>nQ=4x3mD%eWzr;!hQOKkC+w="
    "UUYf@_K{4(zV=3%A9y<HeyGn+@xgDU20+hJ10Ra-UlHfU`$!LU{{8!Z<{ABoQvGAaG3t=hL*Xw{GqImtO7I5UPbT(dOU3?Ad>u0_"
    "6ZOogetZ8tzjnGG*%3E26z@G<*qt<A%)jFgs~Oh0&B5S<7``^(ft0U_I3rIU$bYN6_<qli@%i+b7~cmU-JN_AdiHxqoM-(U@N>T^"
    "0ePy>9N#-0jQrI**hj=&3ErV~Y!Kp+O{=l5^bhkPXA=hk$9#Gi_4BzaATGJRAM$tN74}E`YdrMp!8eFQ{_IThXbc!cc`&2eTJU;@"
    "VZFlq@gq7m-sy!nV$yur`FCr>?+jc7ymj*<_@j{1B&Y7EAvCW}$_?~G=v@o;Kd3t5kW;%M=d}((Kj#m_{%N6o>Da2h4m^f>Pq3e2"
    "P|GCjFSZx;=uLXVPxPzzT;O}~eOu1Oev?^dgV&Hn?BjW0H+^pZ9FO|+x933*7C!*~zAN@+ycP)g{UHi+wy>Mv?}p&I1?!=1{l^(("
    "zwMcmfm3o+#y)LZFC#AawIAYy5~1LA7q16?+K7SRJ5|Sic;>h<;D>#NI(sX*9P045)kU3u=g}t-w<N;ew_KTqx_#so^uCv{&vET|"
    "@Y`<xM0#s{Ki~lNheJPZqvJy44L-#_+~&Cb;14%`dJz2|#-2mJk7wZZG?{_>>6?i@0_}=m|Mu<YP<Q{^m(L~rxcnt>(1Yy2O#^?%"
    "=O6WeTzA+7Ty^Ft@Yj!z5r-ez1i$dsM#z1?n)E&Aqj=~=!EfP5Hl0D=lRv+~_sqO>Y)%P>|ERWrMz?VYc!)P{gKrgd0da}j0DQz2"
    "r+}l56rj=cD*^o&G7$0ms%YS_=&`u(DP0ivzs`qvXXpFCZT>TW=Te5#dG*r05Z@2HO8Vq1EJgZYjo%Nuy&;W8>*P(~r~FOGZrZgE"
    "fQLP$5Arj4lF<*O+a2gpIQrE&5sT4Jq-7JxPeeoL$^Ef$f<FiUpEefta=8(&8aYy+7mG@#fCs+_eZK62xbD-Y(5pG)&w_{46mdYz"
    "f8a;M8)Kin<|PrAuiFbe(=QGC&Q5ED^_CBBf{!>X6#0ekGqG>|z)aw-CjQ_#{Q|#Zyc-O=@V-BIYxP1Y-{cPVhunP_K=HHP8+<>b"
    "UncCp&`kK1?(kbqhfws}m?!Yiz(ClS?gE#Mg&#5I`5>?DGokO_2cjR`=fGFGnTftj-}#VVnC64unj46IHj{#Pp&wr$`V7qtMxUCI"
    "KC(mZbodjaRVJRNe`uV<uYK-^I3m`E(HDw%dUGiLJ`nvCjdJ*&HmNU;d-X-WYuqx_jg)SR<H!_@>puX`Fn4L_!<bO=7y6<)=noXo"
    "mGr{w8Ax($9-oT*Rq#>to2!+bMtfwoqR2x<g0DMZ6!fKmKlGx;Y4oK_Zvk9+XBPTE++G8_-{2_nR#S39zMnjX|8eqwzY`Ti*Rcu&"
    "AU}3ElKhE%rA2P|-^qlB?6N+@4b9+}t>6mi7Zuh5c>HW%^tF3k0sFC^IgI>BD}i%wP6i)wX(aS0rZ4!wyAC7YQXBY4D-(+S>z4;X"
    "4&MzX{OEQF0M2e7h(18?i1^`SfAAzjGtOdPOY}`~vxfqwTxo&)+4&&4A9q^|=)>U(z&)WAkPnEj0QtP00lB^azL)lTGI(UogGr7}"
    "I}&y}JqU5##=~TXjgS_h=$|O=t7!n{$v+dgX=@PTyL^G@my#Xw?;Pq2JiDU>c)JaQvmkF4i8$nTB>E~|nT)ugj6dScJBNYOa)qJ~"
    "*GGYfo7(x&^~|b%N3p-DpV%+AFZ@nw2K2Zh^u>8QfbzUfEBxN{_idH<wY|l-9{Z>n6QMVHpxAdX7InC0^C;-gDe?WKqCU4rMfi~@"
    "#3gRB;3dvR-0!B1z}WH!<g+G(LqB3~Kws()#kl1Y(if+9BKn}18=)_aktZ_y?E^1yu8zKq$7&*v)VnbJNWct=hpjFmKl^!KIyMq2"
    "65cX5L?N$R-5+w_`4sw^4lYQa>%+f6KCG6=e|75#etf_h^jj={6!!JVK;W^KUxEkucqHP9-d7Npx2*;IweKFzANn3}ZjGA2>z5yq"
    "yxTXgAfEjM`r~x;hdgatkI%Dq$9W?fgTLqB9P(ZfxJ%E4J~eJsVH|(E6#T<dr64!0R-)g<qQSr|+EDnJRV$!hS??i!tyu*5xpjL_"
    "f_Fa%_-o`j;Igg@z+3No5d6T4kATOTcSe4||0?>`l`aZ=(s?HOcGhg*Ltl{J(BFDsGV-hKE+bDcF(2gmtEq?&>@L_}yV-ceeZ~{;"
    "m@CgZ06y$-=;P0S@4-I1U*q?`PC$Q`;tTOSdq0AH?caiZlphX*U0Bs0Jm_j2aJ?@#0XM$Y40VprFT#(fw%;ZBkFFSve!yGCP>jxe"
    ")TP@S48-LN&@amlN8Ij~$NIN%Q1Aigi8y{J`k|PQGO&K~K?{5@{0rz&k7bC@+v0O;`)l}*=t=0i82JU(3lqyDpOxGk{d}*Lhkk5-"
    "jeJp`N$@9iCZYesq<zR2<>>=|Vva*TxIcLBc1UCBNB+|Ecei67_>&6d;V&K>K>jxm`0&=)8^~{MTnD+$nhSZ`W|z^=u|^i;!P+Aa"
    "V0R9u(d<(K{$l7x@aXsUfjuw37WgFfci@L@B@pL2OMttoY=T_xzY2Wt!#VW-nTdWoTK*{Dr#;2c|MuxJyx&>BAs*@VA?)JpqUd)x"
    "xFL9>wNJpVZg~WL@uIWvCyURa4^HEz;J3EP4ZeJH(dTyz`UPA4TB0B0^a#Yi@iDls&t{>|&|Snk+9d2lZCvxA?{HQf>o!Zvqd!}L"
    "W9W<e{0?~Z8`quzPdXIOH?Kaf|6K|A$=JT=XEx&|@)XBQ(RqzaACjN4zBmg!8gdcqmHRJ3Uk^@!9t4FQ!~W^(!BZ|#2m8s--U>f;"
    "@)F@dWAuI4hhbTP*COVKx(890x2Z1jSjWDCeNRC=Y=5#7IHpP@t)p9Y`-<@}jYiwPw61RaCz96H_1TlFV?E(8)ti_n4^thMofL$1"
    "lK1u^&UxJe>m|LQA7=j`#3$o3kPn(2ggE^<;&81>B-TG39tJKNS^@bi|GrrFo1FnWe=q}fzo;MV>QLYvy;&sYy?qAcWn=~9ZzsWD"
    "J1_gv`*yzl19|y8KC!QT3*hkdNUV3=-GM$qSptzioeg}XznqMGPphCO0xto#j}bT}1M6*$lYu^rLvbBrpm_f$f{=$ThP<nmKNR_("
    "Lcme>@X5eA9|*jadlvLHe<=J;ULV<A=a&quPb_bNcxBn-6!bsHK)zs03-n!`RsnHQAK)va$7JB1AsGjdw`u`BZjI+L{yH2h{*Jy9"
    "<M3Sejtt7X>fwFC!@tu4{Y@twhQDcb81vy&NWwZ`5c)(Mi^MwNkPP%IX*?Ny8Se$5UuDCfE#h<dkJ!HGH#RYn_<q*DAn@)_Pez{i"
    "LIvdU9#sHN`ZN-9f35}WPD#Iu=vQ_a>m3=9=wtCR2>n`T1tES3iA4Ucc?;feNUhuD`t^EpeF@i}c)VL>KLkVBXUF4RDZDD)e~0Vi"
    "Em7C#?E}g@tIBd+FYmjqsk{=dU*WoWKbddB^<7-w;`O&GBiGe4|3nk(eNIuiZdYSIm8f^|>LIf#^&na0Iu2eB<9dB>-yQKiTd6<d"
    "`gX2|;(B21mcpC5sqDwj^@{28zBF9N<?%p7J&7yoQuGVTzUi+N{$_++FUb4;a{ZULpVTFJ|2tlX^!R$`<@!yp-YG%h#mC8YwBG)1"
    "&t*OV?_2Kmy-HHnDGw`ky)R|Hp|{^uqP(u?@rQX->eCHnzh~~tz<nEBg@^8_dZ6q-<?%$gj*<02sb^$<mBo6X)Nye=JMYiK`%SZc"
    "uwFR4|1{S@3O(Wd+r2tS)(7qj!S!JLcX4bLSM-SaT)f|+&==kx$jPnf6|bwODLhr~|5iw@uhi820w2io@9m%FD*It7ylHQL4~2Kj"
    "`(H6{Pu+ip>++cw$-GpqH)Gxw*Qas)q*qVPeAwGEFP-<(;^*`B<xP<LxoMY_I?$)`eewFdtL%@>bpm?k|KJZs$n|d?k1ay(OTp{r"
    "-hQ$FgFmRNd~}6x>-BA@BY*Dg_gP!6`(>U8>xolC_8+$JA6hlJKHgRQhO77uvw|%D-u`b2AB+8n%lt*9e~K;SU*md7@jc#OQ7a|a"
    "Q=0t!qH;aHDMsd%bA7&+ORk&r>Lg#t^1<~eyq}V1$CdrRy?xYQDfaiNvhUDSWgl|xYmqMdA?}00^(!o=ybqw)r}2@pkK7fdpGBh5"
    "r}3HGH-hz+>!Pjm%0BPx$8O7h%HwB6%YKyWbh(a{>sXlw=Iv9)b|PG^YiE9dw@=qYd9>@wdZ702v6T8=?swv<dQw~VH-=IV?<l+x"
    "?HyT9xZeZ!Lt$P!<1L$cC0r*g^v4l;<0$)3d-^2&kFD^i9HqXP@fv@R^^N@%*Xeuv%T-YPk?<=H?>i^_i(5p-XUyyH_W2d{`yMZn"
    "{f(k0T$jsyIhKEqH=7{WAA3A^uJhvh)muuve2TnJGV|!UE>{<IPTCdO{&Sr%^Fg@Ip79X(8)6;`??>$Q>tX#6`e!M9hS)AJFW2Kg"
    "zf$%M`&-#>GgjflFb_-J2QgNu13oIB*W+W?mHhzQeO)<b|6(ff1^Wk6)gR?N{5_ZTg7t)P7xS!GPpZiHi~R}L&%3M#EdPo>(S+Y{"
    "{A9f4Ds}eE({dF5p)2z3@#2NQ&=fr7_19uvJJ-L9xP$j&^y(I0%Y6u3?hhg262>>){@&tqPu`Q1{X$>Lym~|78#5m9>bRADJB;sH"
    "{yd)9Yng|^d{q60+`o(E+1nTOZyD#f3SWTl-{W)KllN2gc#cW39(#PPYck&C=jVPG9uNJ3EcdMc%)|2b8H`u@FbRF}c!G-DtNcN&"
    "?*G733O__w&ZovFo<6-J$0NKSs4o16%D2)K{V-L2yJF{!axzY`OUt|yy@VW(TOtl{nO7w6kS#{ukJrOT%tLu4^K~qBe{TL>g3MF$"
    "?0vEvcY3_7D1{Ht`|t97Br13&S=n!q`&h)weMA`Nc>M}onSaZ2$NUMef7U7a{2t!AEaw}nvr3<XbFyA?UpcS8#A_Kh+RO_+CF>dU"
    "^_e&5@n4?E{1@)W#C}B&ll_<1k043eU-X_l-yDzpFMh45=mF=mG)4Y(g`ch|xQqP-<1Sl?KV0q)R9)5&N8zJ;c`Me33bMao{;r|q"
    "tJpuVo>)q}!F+S>(`7RrVm(pzbLV}+xj%!qkLxR$&&mA?Y^6T{zZW%s<?Z8nSB~TLTQW}a_D4^b`$4dM)$hvn^XA{OAMtQfs?s;="
    "w9L=;;?z3|U+<}My@xVim-~u1N94SLc2(ZD**qfSEM3{hl<}XppXV{9Z~J4p{|N7k?#(aDeX5+#^Y%$ikn<kApFGD$+<(YY;twyb"
    "uB+fBC6B8q{9FFJuEZyX!rRqV|4~!<T;;JV`CdcCSC!;Af=BMR<;Ce0Wc^`$rTNKmI{O#1gdCUiG3R}~J}U*3ycO?%AV%(=$ok>!"
    "<D4e<$@2QwWy*O_Z-4tFnTMxE$@ZV)Nj`Q&T%+HX{eh+U5x$<cfB17b?pEUgk6-^#=GSvyDfY9B`#fGO=cUfd`6rH(nMcC-(({W^"
    "GVfJad7d%y{c!xQ3IC%DJH!22v=g%bwiW;B^-*$Ueyy$c5mow*I*R_NeSug%Y=x(*E60}7PmALZTj@8)`CqGwtPiF#>dL<Cni_AH"
    "ll6f6{&K%xN9j{%EBe53w!v`-=VMFC{)Ip1INRX3gX4%C%HMOz@rj3v_;cplrz`oERK?$@eW_R;#Cf<6826vkIlg%=_ZMM#WgO?l"
    "y=k()VLamXsp9>@MI6ulLYVK*yemiP-@wnyc|4Cl952T=j6c2oR<0?1AXAjSJ&Zru?%YxOVZ_V(h?wVPJj(m3vtMy*|9AVZ_T#mc"
    "{yvTpPcVMszHy#>SCaEZwz6KqakDqF-0^w@uRm~~UeS-i5ptk$pDazz^LTwL*pI%F^OIhlK2Gjq$#$Categk;`jN2z<~(br{Jy+?"
    "APGu;=wyX=cT2|Aybq7JUyib$Ilp&~cdZk0KWlGa^Jsa!fuGysy>njnto;0@+E0t)j9W@SJEhOHb5vfJ^!h#XIz_6iC)`iP+xLfY"
    "-a%zOk@4YUIbLR5%e;2pZ^+##|Bi8%l`gMedVM2P<v5A^G_#&+FZ}->3cEF7"
)


def load_embedded_data() -> np.ndarray:
    """Decode and verify the embedded loss trajectories."""

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
    return np.frombuffer(raw, dtype=_DATA_DTYPE).reshape(_DATA_SHAPE).copy()


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

    for rank_index, rank in enumerate(RANKS):
        raw_runs = data[rank_index, metric_index]
        runs = smooth_runs(raw_runs, ema_alpha)
        mean, error = summarize_runs(runs, uncertainty)
        iterations = np.arange(1, len(mean) + 1)
        color = COLORS[rank]
        marker = MARKERS[rank]

        axis.plot(
            iterations,
            mean,
            color=color,
            marker=marker,
            markevery=error_every,
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=f"r{rank}",
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
    parser = argparse.ArgumentParser(
        description=(
            "Plot embedded r4/r8/r16 mean losses with standard-error bars."
        )
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
    output_dir = Path(__file__).resolve().parent

    plot_metric(
        data=data,
        metric_index=TRAIN_LOSS_INDEX,
        output_path=output_dir / "train_loss_r4_r8_r16.pdf",
        ylabel="Lower-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )
    plot_metric(
        data=data,
        metric_index=VALIDATION_LOSS_INDEX,
        output_path=output_dir / "validation_loss_r4_r8_r16.pdf",
        ylabel="Upper-level Loss",
        ema_alpha=args.ema_alpha,
        uncertainty=args.uncertainty,
        error_every=args.error_every,
    )
    print("Plots generated successfully.")


if __name__ == "__main__":
    main()

