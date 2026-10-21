/*
 * Simulate with ns-3 a scenario exported by ECLYPSE+NET (see scenario.py).
 *
 * The program reads from --input the files written by export_for_ns3:
 *
 *   nodes.csv    name,role
 *   links.csv    u,v,bw_uv_mbps,bw_vu_mbps,delay_s,queue_uv,queue_vu
 *   routes.csv   node,destination,next_hop
 *   packets.csv  id,src,dst,size_bytes,send_time_s
 *
 * and builds the same network:
 *
 *   - every link is a point-to-point channel, with the bandwidth of each direction
 *     as data rate of the sending device and the propagation delay of the link;
 *   - every device has a DropTail queue. In ECLYPSE the capacity of a link counts
 *     the packet being transmitted, while an ns-3 device transmits one packet
 *     outside its queue, so the queue holds capacity - 1 packets. The queue disc
 *     of the traffic control layer is removed, so packets go straight to the
 *     device queue; when it is full, the traffic control layer drops them (flow
 *     control), which is equivalent to DropTail;
 *   - routing is static: every node gets one host route per destination, towards
 *     the next hop that ECLYPSE used, so packets follow the same paths;
 *   - hosts do not forward packets.
 *
 * Every packet is sent over UDP at its send time, from its source host to its
 * destination host. Its payload carries its ID and is size - 30 bytes long, so that
 * with the UDP (8), IPv4 (20) and PPP (2) headers it occupies size bytes on the
 * links, as in ECLYPSE. The simulation runs until no event is left.
 *
 * The program writes to --output:
 *
 *   received.csv  id,time_s       one row per packet received by its destination
 *   stats.csv     key,value       counts and wall-clock times
 *
 * Usage, after copying this file in the scratch folder of ns-3:
 *
 *   ./ns3 build eclypse_scenario
 *   ./ns3 run --no-build "eclypse_scenario --input=DIR --output=DIR"
 */

#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/traffic-control-module.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ns3;

namespace
{

constexpr uint16_t kPort = 9;
constexpr uint32_t kWireOverhead = 30; // PPP (2) + IPv4 (20) + UDP (8)

struct LinkEnd
{
    uint32_t interface;  // interface of the sending node on the link
    Ipv4Address nextHop; // address of the receiving node on the link
};

std::ofstream g_received;
uint64_t g_receivedCount = 0;
uint64_t g_queueDrops = 0;
uint64_t g_ipDrops = 0;

std::vector<std::vector<std::string>>
ReadCsv(const std::string& path)
{
    std::ifstream in(path);
    NS_ABORT_MSG_IF(!in, "Cannot open " << path);
    std::vector<std::vector<std::string>> rows;
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            continue;
        }
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ','))
        {
            fields.push_back(field);
        }
        rows.push_back(fields);
    }
    return rows;
}

void
Receive(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom(from)))
    {
        uint64_t id;
        packet->CopyData(reinterpret_cast<uint8_t*>(&id), sizeof(id));
        g_received << id << ',' << Simulator::Now().GetSeconds() << '\n';
        ++g_receivedCount;
    }
}

void
Send(Ptr<Socket> socket, uint64_t id, uint32_t size, Ipv4Address destination)
{
    std::vector<uint8_t> payload(size - kWireOverhead, 0);
    std::memcpy(payload.data(), &id, sizeof(id));
    Ptr<Packet> packet = Create<Packet>(payload.data(), payload.size());
    socket->SendTo(packet, 0, InetSocketAddress(destination, kPort));
}

// Packets dropped because the queue of the outgoing device is full: by the
// traffic control layer when the device has stopped its queue (flow control), or
// by the device queue itself
void
QueueDrop(Ptr<const Packet>)
{
    ++g_queueDrops;
}

void
IpDrop(const Ipv4Header&, Ptr<const Packet>, Ipv4L3Protocol::DropReason, Ptr<Ipv4>, uint32_t)
{
    ++g_ipDrops;
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string input = ".";
    std::string output = ".";
    CommandLine cmd(__FILE__);
    cmd.AddValue("input", "Folder with the scenario exported by ECLYPSE", input);
    cmd.AddValue("output", "Folder of the results", output);
    cmd.Parse(argc, argv);

    auto setupStart = std::chrono::steady_clock::now();

    // Nodes
    std::unordered_map<std::string, Ptr<Node>> nodes;
    std::unordered_map<std::string, bool> isHost;
    NodeContainer all;
    for (const auto& row : ReadCsv(input + "/nodes.csv"))
    {
        Ptr<Node> node = CreateObject<Node>();
        nodes[row[0]] = node;
        isHost[row[0]] = row[1] == "host";
        all.Add(node);
    }

    InternetStackHelper stack;
    Ipv4StaticRoutingHelper staticRouting;
    stack.SetRoutingHelper(staticRouting);
    stack.SetIpv6StackInstall(false);
    stack.Install(all);

    // Links
    PointToPointHelper p2p;
    Ipv4AddressHelper addresses("10.0.0.0", "255.255.255.252");
    TrafficControlHelper trafficControl;
    std::map<std::pair<std::string, std::string>, LinkEnd> linkEnds;
    std::unordered_map<std::string, Ipv4Address> hostAddress;
    uint64_t nLinks = 0;
    for (const auto& row : ReadCsv(input + "/links.csv"))
    {
        const std::string& u = row[0];
        const std::string& v = row[1];
        p2p.SetChannelAttribute("Delay", TimeValue(Seconds(std::stod(row[4]))));
        NetDeviceContainer devices = p2p.Install(nodes.at(u), nodes.at(v));
        const double bandwidths[2] = {std::stod(row[2]), std::stod(row[3])};
        const uint32_t capacities[2] = {static_cast<uint32_t>(std::stoul(row[5])),
                                        static_cast<uint32_t>(std::stoul(row[6]))};
        for (uint32_t i = 0; i < 2; ++i)
        {
            auto device = DynamicCast<PointToPointNetDevice>(devices.Get(i));
            device->SetDataRate(DataRate(static_cast<uint64_t>(bandwidths[i] * 1e6)));
            NS_ABORT_MSG_IF(capacities[i] < 2, "Queue capacities must be at least 2");
            device->GetQueue()->SetMaxSize(
                QueueSize(QueueSizeUnit::PACKETS, capacities[i] - 1));
            device->GetQueue()->TraceConnectWithoutContext("Drop",
                                                           MakeCallback(&QueueDrop));
        }
        Ipv4InterfaceContainer interfaces = addresses.Assign(devices);
        addresses.NewNetwork();
        trafficControl.Uninstall(devices);

        uint32_t ifU = nodes.at(u)->GetObject<Ipv4>()->GetInterfaceForDevice(devices.Get(0));
        uint32_t ifV = nodes.at(v)->GetObject<Ipv4>()->GetInterfaceForDevice(devices.Get(1));
        linkEnds[{u, v}] = {ifU, interfaces.GetAddress(1)};
        linkEnds[{v, u}] = {ifV, interfaces.GetAddress(0)};
        if (isHost.at(u))
        {
            hostAddress[u] = interfaces.GetAddress(0);
        }
        if (isHost.at(v))
        {
            hostAddress[v] = interfaces.GetAddress(1);
        }
        ++nLinks;
    }

    for (const auto& [name, node] : nodes)
    {
        Ptr<Ipv4> ipv4 = node->GetObject<Ipv4>();
        if (isHost.at(name))
        {
            ipv4->SetAttribute("IpForward", BooleanValue(false));
        }
        ipv4->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext(
            "Drop",
            MakeCallback(&IpDrop));
        node->GetObject<TrafficControlLayer>()->TraceConnectWithoutContext(
            "TcDrop",
            MakeCallback(&QueueDrop));
    }

    // Routes: the next hops used by ECLYPSE
    uint64_t nRoutes = 0;
    for (const auto& row : ReadCsv(input + "/routes.csv"))
    {
        const LinkEnd& end = linkEnds.at({row[0], row[2]});
        Ptr<Ipv4StaticRouting> routing =
            staticRouting.GetStaticRouting(nodes.at(row[0])->GetObject<Ipv4>());
        routing->AddHostRouteTo(hostAddress.at(row[1]), end.nextHop, end.interface);
        ++nRoutes;
    }

    // Traffic: one sending socket per source host, one receiving socket per
    // destination host
    std::unordered_map<std::string, Ptr<Socket>> senders;
    std::unordered_map<std::string, Ptr<Socket>> receivers;
    TypeId udp = UdpSocketFactory::GetTypeId();
    uint64_t nPackets = 0;
    for (const auto& row : ReadCsv(input + "/packets.csv"))
    {
        const std::string& src = row[1];
        const std::string& dst = row[2];
        if (!senders.count(src))
        {
            senders[src] = Socket::CreateSocket(nodes.at(src), udp);
            senders[src]->Bind();
        }
        if (!receivers.count(dst))
        {
            Ptr<Socket> socket = Socket::CreateSocket(nodes.at(dst), udp);
            socket->Bind(InetSocketAddress(Ipv4Address::GetAny(), kPort));
            socket->SetRecvCallback(MakeCallback(&Receive));
            receivers[dst] = socket;
        }
        Simulator::Schedule(Seconds(std::stod(row[4])),
                            &Send,
                            senders.at(src),
                            static_cast<uint64_t>(std::stoull(row[0])),
                            static_cast<uint32_t>(std::stoul(row[3])),
                            hostAddress.at(dst));
        ++nPackets;
    }

    g_received.open(output + "/received.csv");
    g_received << "id,time_s\n" << std::setprecision(17);

    auto runStart = std::chrono::steady_clock::now();
    Simulator::Run();
    auto runEnd = std::chrono::steady_clock::now();
    Simulator::Destroy();
    g_received.close();

    std::chrono::duration<double> setup = runStart - setupStart;
    std::chrono::duration<double> run = runEnd - runStart;
    std::ofstream stats(output + "/stats.csv");
    stats << std::setprecision(17) << "key,value\n"
          << "nodes," << all.GetN() << '\n'
          << "links," << nLinks << '\n'
          << "routes," << nRoutes << '\n'
          << "sent," << nPackets << '\n'
          << "received," << g_receivedCount << '\n'
          << "queue_drops," << g_queueDrops << '\n'
          << "ip_drops," << g_ipDrops << '\n'
          << "setup_time_s," << setup.count() << '\n'
          << "run_time_s," << run.count() << '\n';
    return 0;
}
