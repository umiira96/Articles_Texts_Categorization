"?:
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1    ??AA    ??AasjNp???isjNp????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff??@9ffff??@Affff??@Iffff??@a??6f???iP?A??,???Unknown?
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1??????@9??????@A??????@I??????@a.?;???u?i,PD?X???Unknown
iHostWriteSummary"WriteSummary(1     ??@9     ??@A     ??@I     ??@a?7??<r?i?&?}???Unknown?
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1???????@9???????@A???????@I???????@a???}?gq?i&?۟???Unknown
?HostResourceGather")sequential_7/embedding_7/embedding_lookup(1?????[?@9?????[?@A?????[?@I?????[?@a?P??	q?iP?	?????Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1?????E?@9?????E?@A?????E?@I?????E?@a?[	s?p?i??/?????Unknown
?	HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(133333O?@933333O?@A33333O?@I33333O?@a,?$?~Np?ir??-r???Unknown
g
HostMul"Adam/Adam/update/mul_4(1?????x?@9?????x?@A?????x?@I?????x?@a?}x? ?n?i?f?.%#???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1??????@9??????@A??????@I??????@a??|??Jn?i??pA???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1?????̉@9?????̉@A?????̉@I?????̉@a6?0??j?i??7?O\???Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1????̨?@9????̨?@A????̨?@I????̨?@a?{ec[?j?iQ?;
w???Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333'?@933333'?@A33333'?@I33333'?@aj$1?a?iou?M?????Unknown
gHostMul"Adam/Adam/update/mul_5(1     0@9     0@A     0@I     0@a?f?>`?i[???&????Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1?????A~@9?????A~@A?????A~@I?????A~@a???)*?_?iЬ???????Unknown
kHostUnique"Adam/Adam/update/Unique(1?????iw@9?????iw@A?????iw@I?????iw@a4r?3cX?i?e?;????Unknown
fHost_Send"IteratorGetNext/_13(1?????s@9?????s@A?????s@I?????s@a?d????S?i?=?????Unknown
mHostRealDiv"Adam/Adam/update/truediv(1?????o@9?????o@A?????o@I?????o@a?G?z?2P?i@??M????Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(133333n@933333n@A33333n@I33333n@aũ?*CO?i?C??????Unknown
gHostMul"Adam/Adam/update/mul_1(133333Cf@933333Cf@A33333Cf@I33333Cf@a!???0G?iu?H;?????Unknown
?Host_Send"-sequential_7/embedding_7/embedding_lookup/_25(1??????d@9??????d@A??????d@I??????d@a???$??E?i?'?3+????Unknown
gHostMul"Adam/Adam/update/mul_2(1fffff?c@9fffff?c@Afffff?c@Ifffff?c@aD??b9gD?i]?*E????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1?????)a@9?????)a@A?????)a@I?????)a@aķ-??A?iN<?&?????Unknown
gHostMul"Adam/Adam/update/mul_3(133333?_@933333?_@A33333?_@I33333?_@a??"??@?i?????????Unknown
gHostAddV2"Adam/Adam/update/add(1fffff?\@9fffff?\@Afffff?\@Ifffff?\@a????=?iEe?Ƙ????Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1      U@9      U@A      U@I      U@a??S	??5?i????T????Unknown
eHostMul"Adam/Adam/update/mul(1fffffFT@9fffffFT@AfffffFT@IfffffFT@a?м?5?i?)???????Unknown
?Host_Recv"Egradient_tape/sequential_7/embedding_7/embedding_lookup/Reshape_1/_28(1??????R@9??????R@A??????R@I??????R@a?0A?:?3?i?q?;k????Unknown
dHostDataset"Iterator::Model(1?????P@9?????P@A     ?I@I     ?I@a?????*?iQAEd????Unknown
eHost
LogicalAnd"
LogicalAnd(1?????I@9?????I@A?????I@I?????I@aI??A?*?ia	??????Unknown?
^ HostGatherV2"GatherV2(1??????B@9??????B@A??????B@I??????B@a???u?_#?i??`??????Unknown
i!Host_Recv"Adam/ReadVariableOp/_2(1      A@9      A@A      A@I      A@a???=?!?i??14????Unknown
?"HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff??@9fffff??@A333333:@I333333:@a<ÖH?J?i7A\??????Unknown
?#HostVariableShape"Egradient_tape/sequential_7/embedding_7/embedding_lookup/VariableShape(1??????6@9??????6@A??????6@I??????6@a?:?w߿?i??W??????Unknown
?$HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      7@9      7@A?????0@I?????0@aQ?{\>??i??J?)????Unknown
f%Host_Send"IteratorGetNext/_11(1ffffff-@9ffffff-@Affffff-@Iffffff-@ar?֟?i?0?????Unknown
s&HostDataset"Iterator::Model::ParallelMapV2(1??????)@9??????)@A??????)@I??????)@a6?0??
?i*??????Unknown
`'HostGatherV2"
GatherV2_1(1??????(@9??????(@A??????(@I??????(@a!?Z?ܟ	?i?q?/v????Unknown
x(HostStridedSlice"Adam/Adam/update/strided_slice(1      '@9      '@A      '@I      '@a??4??i?U?????Unknown
Y)HostPow"Adam/Pow(1??????&@9??????&@A??????&@I??????&@a ??k???iO0.4????Unknown
p*Host_Recv"Adam/Cast_2/ReadVariableOp/_6(1??????"@9??????"@A??????"@I??????"@a?0A?:??iT9i??????Unknown
?+Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1??????@9??????@A??????@I??????@a???>u ?i?adX?????Unknown
x,HostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????yP@9?????yP@Affffff@Iffffff@aNtJ??>i??b?????Unknown
?-HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a?F??ؿ?>i?3-=????Unknown
?.HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a?:?w߿?>iE#Ӭl????Unknown
l/HostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a?:?w߿?>i??,?????Unknown
[0HostSub"
Adam/sub_2(1ffffff@9ffffff@Affffff@Iffffff@a8/?i?*?>iW沁?????Unknown
[1HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@a*??=?>i%?.??????Unknown
[2HostSub"
Adam/sub_3(1333333@9333333@A333333@I333333@a(ɽ???>i??,????Unknown
[3HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a?KV+??>iY-$????Unknown
?4HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff:@9ffffff:@A333333@I333333@a?E??.U?>i	?[V@????Unknown
?5Host_Send"Hgradient_tape/sequential_7/embedding_7/embedding_lookup/VariableShape/_9(1333333@9333333@A333333@I333333@a?;f??*?>ioo??X????Unknown
p6Host_Recv"Adam/Cast_3/ReadVariableOp/_8(1??????@9??????@A??????@I??????@ao7*.??>i??? o????Unknown
t7Host_Recv"!sequential_7/embedding_7/Cast/_24(1??????@9??????@A??????@I??????@ao7*.??>i?˧??????Unknown
]8HostCast"Adam/Cast_1(1??????@9??????@A??????@I??????@aa5????>i??4+?????Unknown
a9HostRealDiv"Adam/truediv(1??????@9??????@A??????@I??????@a+-?8<U?>icq??????Unknown
[:HostSqrt"	Adam/Sqrt(1333333??9333333??A333333??I333333??a?E??.U?>iaD??????Unknown
Y;HostMul"Adam/mul(1333333??9333333??A333333??I333333??a?E??.U?>i_????????Unknown
n<Host_Recv"Adam/Cast/ReadVariableOp/_4(1????????9????????A????????I????????a?A?#???>i???*?????Unknown
?=Host_Recv"Cgradient_tape/sequential_7/embedding_7/embedding_lookup/Reshape/_34(1????????9????????A????????I????????a?A?#???>i?%??????Unknown
[>HostSub"
Adam/sub_1(1      ??9      ??A      ??I      ??a?=?????>i?? ?????Unknown
a?HostIdentity"Identity(1????????9????????A????????I????????a+-?8<U?>i+??*?????Unknown?
Y@HostSub"Adam/sub(1ffffff??9ffffff??Affffff??Iffffff??a{9H_5U?>i?????????Unknown*?8
uHostFlushSummaryWriter"FlushSummaryWriter(1ffff??@9ffff??@Affff??@Iffff??@a(??????i(???????Unknown?
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1??????@9??????@A??????@I??????@a0n_?4??i?͟?1???Unknown
iHostWriteSummary"WriteSummary(1     ??@9     ??@A     ??@I     ??@anH?:L??i?m??M????Unknown?
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1???????@9???????@A???????@I???????@a<???;<??i??3?????Unknown
?HostResourceGather")sequential_7/embedding_7/embedding_lookup(1?????[?@9?????[?@A?????[?@I?????[?@a['g5xĥ?ii?0YM???Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1?????E?@9?????E?@A?????E?@I?????E?@a?`?1???i?i0Ķ???Unknown
?HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(133333O?@933333O?@A33333O?@I33333O?@a?_D8?Ԥ?i꯴^????Unknown
gHostMul"Adam/Adam/update/mul_4(1?????x?@9?????x?@A?????x?@I?????x?@a?Z??????i?%???.???Unknown
}	HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1??????@9??????@A??????@I??????@a#J?lY??i?ǔ?qd???Unknown
}
HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1?????̉@9?????̉@A?????̉@I?????̉@a?PN?*??iԬY?w???Unknown
?HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1????̨?@9????̨?@A????̨?@I????̨?@a?<1&???i???(C????Unknown
?HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333'?@933333'?@A33333'?@I33333'?@aN	U?fӖ?i?h_?>???Unknown
gHostMul"Adam/Adam/update/mul_5(1     0@9     0@A     0@I     0@aGNM?.???i`?a??????Unknown
gHostSqrt"Adam/Adam/update/Sqrt(1?????A~@9?????A~@A?????A~@I?????A~@a??B?!??i??CW?????Unknown
kHostUnique"Adam/Adam/update/Unique(1?????iw@9?????iw@A?????iw@I?????iw@a??!?'??i?q??????Unknown
fHost_Send"IteratorGetNext/_13(1?????s@9?????s@A?????s@I?????s@aX???N??i??	?g???Unknown
mHostRealDiv"Adam/Adam/update/truediv(1?????o@9?????o@A?????o@I?????o@a?A??G???i??'?????Unknown
?HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(133333n@933333n@A33333n@I33333n@a]?T|???i??Ul
???Unknown
gHostMul"Adam/Adam/update/mul_1(133333Cf@933333Cf@A33333Cf@I33333Cf@a??@s??}?i?oD?E???Unknown
?Host_Send"-sequential_7/embedding_7/embedding_lookup/_25(1??????d@9??????d@A??????d@I??????d@a?8gU?{?i?]??\}???Unknown
gHostMul"Adam/Adam/update/mul_2(1fffff?c@9fffff?c@Afffff?c@Ifffff?c@a???òz?iy-T~????Unknown
?HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1?????)a@9?????)a@A?????)a@I?????)a@a?{b??v?io??+????Unknown
gHostMul"Adam/Adam/update/mul_3(133333?_@933333?_@A33333?_@I33333?_@a??Ezu?iL?}yZ	???Unknown
gHostAddV2"Adam/Adam/update/add(1fffff?\@9fffff?\@Afffff?\@Ifffff?\@a???2?s?i??iz/???Unknown
{Host_Send"(Adam/Adam/update/AssignSubVariableOp/_36(1      U@9      U@A      U@I      U@a:N????k?if)?KlK???Unknown
eHostMul"Adam/Adam/update/mul(1fffffFT@9fffffFT@AfffffFT@IfffffFT@a?????j?i???2gf???Unknown
?Host_Recv"Egradient_tape/sequential_7/embedding_7/embedding_lookup/Reshape_1/_28(1??????R@9??????R@A??????R@I??????R@a2?mi?i?K?k???Unknown
dHostDataset"Iterator::Model(1?????P@9?????P@A     ?I@I     ?I@a?I>?"a?iC??????Unknown
eHost
LogicalAnd"
LogicalAnd(1?????I@9?????I@A?????I@I?????I@an?]?Ǫ`?i???j8????Unknown?
^HostGatherV2"GatherV2(1??????B@9??????B@A??????B@I??????B@a`???K?X?iE념?????Unknown
iHost_Recv"Adam/ReadVariableOp/_2(1      A@9      A@A      A@I      A@a?ݒ?<?V?i?4?.?????Unknown
? HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1fffff??@9fffff??@A333333:@I333333:@aʉ???nQ?i??i??????Unknown
?!HostVariableShape"Egradient_tape/sequential_7/embedding_7/embedding_lookup/VariableShape(1??????6@9??????6@A??????6@I??????6@a?MiWN?iL?H5????Unknown
?"HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1      7@9      7@A?????0@I?????0@a?Q??lE?i??(q?????Unknown
f#Host_Send"IteratorGetNext/_11(1ffffff-@9ffffff-@Affffff-@Iffffff-@a[PШ??C?i?*_t????Unknown
s$HostDataset"Iterator::Model::ParallelMapV2(1??????)@9??????)@A??????)@I??????)@a?PN?*A?il>? ?????Unknown
`%HostGatherV2"
GatherV2_1(1??????(@9??????(@A??????(@I??????(@a??͂!^@?i??6??????Unknown
x&HostStridedSlice"Adam/Adam/update/strided_slice(1      '@9      '@A      '@I      '@ap??A4?>?i%???????Unknown
Y'HostPow"Adam/Pow(1??????&@9??????&@A??????&@I??????&@a????>?i*8?Ml????Unknown
p(Host_Recv"Adam/Cast_2/ReadVariableOp/_6(1??????"@9??????"@A??????"@I??????"@a2?m9?i??ی????Unknown
?)Host	_HostSend"Fcategorical_crossentropy/softmax_cross_entropy_with_logits/Shape_2/_17(1??????@9??????@A??????@I??????@a????q5?i???-????Unknown
x*HostDataset"#Iterator::Model::ParallelMapV2::Zip(1?????yP@9?????yP@Affffff@Iffffff@ai?F:4?i?C??????Unknown
?+HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1??????@9??????@A??????@I??????@a?P?Z]2?i? /? ????Unknown
?,HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9??????@A??????@I??????@a?MiW.?i??U ?????Unknown
l-HostIteratorGetNext"IteratorGetNext(1??????@9??????@A??????@I??????@a?MiW.?i'D|q?????Unknown
[.HostSub"
Adam/sub_2(1ffffff@9ffffff@Affffff@Iffffff@a?k?)|(?in%4S????Unknown
[/HostPow"
Adam/Pow_1(1ffffff@9ffffff@Affffff@Iffffff@aV3Y??%?i???a?????Unknown
[0HostSub"
Adam/sub_3(1333333@9333333@A333333@I333333@aP?O?$?i?'???????Unknown
[1HostAddV2"Adam/add(1??????@9??????@A??????@I??????@a?l?ȱ#?i?x?7????Unknown
?2HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1ffffff:@9ffffff:@A333333@I333333@a???"?i???2Y????Unknown
?3Host_Send"Hgradient_tape/sequential_7/embedding_7/embedding_lookup/VariableShape/_9(1333333@9333333@A333333@I333333@aB?V??ilj?-P????Unknown
p4Host_Recv"Adam/Cast_3/ReadVariableOp/_8(1??????@9??????@A??????@I??????@a??VG??i,?6????Unknown
t5Host_Recv"!sequential_7/embedding_7/Cast/_24(1??????@9??????@A??????@I??????@a??VG??i??????Unknown
]6HostCast"Adam/Cast_1(1??????@9??????@A??????@I??????@ah????i?l??????Unknown
a7HostRealDiv"Adam/truediv(1??????@9??????@A??????@I??????@aD?l?k?iA?(ݴ????Unknown
[8HostSqrt"	Adam/Sqrt(1333333??9333333??A333333??I333333??a????i???E????Unknown
Y9HostMul"Adam/mul(1333333??9333333??A333333??I333333??a????i3? m?????Unknown
n:Host_Recv"Adam/Cast/ReadVariableOp/_4(1????????9????????A????????I????????a?4 v?i?^????Unknown
?;Host_Recv"Cgradient_tape/sequential_7/embedding_7/embedding_lookup/Reshape/_34(1????????9????????A????????I????????a?4 v?i?b??????Unknown
[<HostSub"
Adam/sub_1(1      ??9      ??A      ??I      ??a??|???i??سf????Unknown
a=HostIdentity"Identity(1????????9????????A????????I????????aD?l?k?iЏbb?????Unknown?
Y>HostSub"Adam/sub(1ffffff??9ffffff??Affffff??Iffffff??a??????>i     ???Unknown